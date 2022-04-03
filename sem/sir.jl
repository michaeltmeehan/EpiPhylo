using BenchmarkTools
using Plots
using StatsBase
using Distributions
using UnPack
using Random

struct Parameters
    R0::Float64
    γ::Float64
    nagents::Int64
    ninfectives::Int64
    t0::Float64
    tf::Float64
end


struct Point
    t::Float64
    S::Float64
    I::Float64
    R::Float64
end


struct State
    p::Point
    cumulative_incidence::Int64
    iter::Int64
    infectives::Vector{Int64}
end


struct Infection
    infectee_id::Int64
    infector_id::Int64
    time_of_infection::Float64
    time_of_recovery::Float64
    time_of_sampling::Float64
end


mutable struct Outbreak
    parms::Parameters
    cstate::State
    tree::Vector{Infection}
    trajectory::Vector{Point}
end


function getInfectives(tree::Vector{Infection}, t::Float64)::Vector{Int64}
    infectives = Vector{Float64}()
    for i in tree
        if i.infectee_id < 0 || i.time_of_infection > t
            break
        elseif i.time_of_recovery > t
            push!(infectives, i.infectee_id)
        end
    end
    return infectives
end


function simulateOutbreak!(out::Outbreak)::Outbreak
    @unpack R0, γ, nagents, ninfectives, t0, tf = out.parms
    @unpack p, cumulative_incidence, iter, infectives = out.cstate
    @unpack t, S, I, R = p

    N = S + I + R

    while (I > 0.0) && (t < tf) && (iter < 2*nagents)
        iter += 1
        infection_rate = R0 * γ * S * I / N
        recovery_rate = γ * I

        r = rand()
        t -= log(r) / (infection_rate + recovery_rate)
        if r <= infection_rate / (infection_rate + recovery_rate)
            infector = sample(infectives)
            cumulative_incidence += 1
            infectee = cumulative_incidence
            out.tree[infectee] = Infection(infectee, infector, t, Inf, -1.0)
            push!(infectives,infectee)
            S -= 1.0
            I += 1.0
        else
            r_i = sample(1:length(infectives))
            recovered = infectives[r_i]
            r = out.tree[recovered]
            out.tree[recovered] = Infection(r.infectee_id, r.infector_id, r.time_of_infection, t, r.time_of_sampling)
            deleteat!(infectives, r_i)
            I -= 1.0
            R += 1.0
        end
        p = Point(t, S, I, R)
        out.trajectory[iter] = p
    end
    out.cstate = State(p, cumulative_incidence, iter, infectives)
    out.tree = out.tree[1:cumulative_incidence]
    out.trajectory = out.trajectory[1:iter]
    return out
end


function simulateOutbreak(R0::Float64=2.0,
                          γ::Float64=1.0,
                          nagents::Int64=1_000,
                          ninfectives::Int64=1,
                          tspan::Tuple{Float64, Float64}=(0.0, 100.0))::Outbreak

    parms = Parameters(R0, γ, nagents, ninfectives, minimum(tspan), maximum(tspan))
    p = Point(minimum(tspan), nagents - ninfectives, ninfectives, 0.0)
    cstate = State(p, ninfectives, 1, [i for i in 1:ninfectives])
    tree = vcat([Infection(i, 0, 0.0, Inf, -1.0) for i in 1:ninfectives], 
                [Infection(-1, -1, -1.0, -1.0, -1.0) for _ in (ninfectives+1):nagents])
    trajectory = vcat([p], [Point(-1.0, -1.0, -1.0, -1.0) for _ in 2:2*nagents])

    out = Outbreak(parms, cstate, tree, trajectory)

    return simulateOutbreak!(out)
end


function pruneTree(out::Outbreak,
                   t::Float64,
                   π::Float64)::Vector{Infection}
    stree = copy(out.tree)
    infectives = getInfectives(out.tree, t)
    nsampled = rand(Binomial(length(infectives), π))
    sampled = sample(infectives, nsampled, replace=false, ordered=true)
    unsampled = Dict{Int64, Float64}()
    for s in Iterators.reverse(sampled)
        stree[s] = Infection(s, 
                             stree[s].infector_id, 
                             stree[s].time_of_infection, 
                             t,
                             t)
        previous_ancestor = s
        current_ancestor = out.tree[previous_ancestor].infector_id
        while current_ancestor > 0
            if current_ancestor in sampled
                break
            elseif !(current_ancestor in keys(unsampled)) || unsampled[current_ancestor] < out.tree[previous_ancestor].time_of_infection
                unsampled[current_ancestor] = out.tree[previous_ancestor].time_of_infection
            end
            previous_ancestor = current_ancestor
            current_ancestor = out.tree[current_ancestor].infector_id
        end
    end

    for (id, t) in unsampled
        stree[id] = Infection(id, 
                              stree[id].infector_id, 
                              stree[id].time_of_infection, 
                              t,
                              stree[id].time_of_sampling)
        push!(sampled, id)
    end
    return stree[sort(sampled)]
end


function getSampled(tree::Vector{Infection})::Vector{Int64}
    return [i.infectee_id for i in tree if i.time_of_sampling >= 0.0]
end


function getCSD(tree::Vector{Infection}, t::Vector{Float64})::Vector{Vector{Int64}}
    sort!(t)
    sampled = getSampled(tree)
    clusters = Dict(s=>[s] for s in sampled)
    current_time = tree[end].time_of_sampling
    csd_out = Vector{Vector{Int64}}()
    for i in Iterators.reverse(tree)
        current_time = i.time_of_infection

        if current_time < t[end]
            pop!(t)
            csd = Vector{Int64}()
            for cluster in values(clusters)
                push!(csd, length(cluster))
            end
            push!(csd_out, csd)
            if length(t) == 0
                break
            end
        end

        clusters[i.infector_id] = i.infector_id in keys(clusters) ? 
                                    vcat(clusters[i.infectee_id], clusters[i.infector_id]) : clusters[i.infectee_id]

        delete!(clusters, i.infectee_id)
    end
    return csd_out
end


#Random.seed!(3)
out = simulateOutbreak(10.0/3.0, 0.3, 100_000, 1_000);
#for i in out.tree
#    println(i)
#end
#println(out.trajectory)
s = pruneTree(out, 10.0, 1.0);
csd = getCSD(s, [5.0])
#print(s)
#print(csd)

#t = [out.trajectory[i].t for i in 1:length(out.trajectory)];
#I = [out.trajectory[i].I for i in 1:length(out.trajectory)];

#plot(t, I)
