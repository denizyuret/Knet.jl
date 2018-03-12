ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
for p in ("Gym","ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
julia reinforce_discrete.jl

This example implements the REINFORCE algorithm from
`Simple statistical gradient-following algorithms for
connectionist reinforcement learning.`,  Williams, Ronald J.
Machine learning, 8(3-4):229–256, 1992. This example also
demonstrates the usage of the `@zerograd` function for
stopping the gradient flow.
"""
module REINFORCE_DISCRETE

using Gym, ArgParse, Knet, AutoGrad

function predict_linear(w, ob)
    linear = w["w"] * ob .+ w["b"]
    return linear
end

function sample_action(linear)
    linear = Array(linear)
    probs = exp.(linear) ./ sum(exp.(linear), 1)
    c_probs = cumsum(probs)
    return indmax(c_probs .> rand())
end

@zerograd sample_action(linear)

function play(w, ob)
    linear = predict_linear(w, ob)
    action = sample_action(linear)
    return action, linear
end

function play_episode(w, env, o)
    ob = reset!(env)
    rewards = Float32[]
    linears = Any[]
    actions = Array{Int,1}()
    total = 0

    for t=1:env.spec.max_episode_steps
        ob_inp = convert(o["atype"], reshape(ob, size(ob, 1), 1))
        action, linear = play(w, ob_inp)
        push!(linears, linear)
        ob, reward, done, _ = step!(env, action-1)

        total += reward

        push!(rewards, reward)
        push!(actions, action)

        if o["render"]
            render(env)
        end

        if done
            break
        end
    end
    return linears, actions, rewards, total
end

function loss(w, env, o; totalr=nothing)
    linears, actions, rewards, total = play_episode(w, env, o)
    totalr[1] = total

    y = zeros(size(w["w"], 1), length(actions))

    for i=1:length(actions)
        y[actions[i], i] = 1.0
    end

    actions = convert(o["atype"], y)

    discounted = discount(rewards; γ=o["gamma"])
    discounted = discounted .- mean(discounted)#mean R as a baseline

    discounted = convert(o["atype"], reshape(discounted, 1, size(actions, 2)))

    logps = sum(logp(hcat(linears...), 1) .* actions, 1)
    -sum(logps .* discounted) ./ size(actions, 2)
end

lossgradient = grad(loss)

function init_params(input, output, atype)
    w = Dict()
    w["w"] = xavier(output, input)
    w["b"] = zeros(output, 1)

    for k in keys(w)
        w[k] = convert(atype, w[k])
    end

    return w
end

function discount(rewards; γ=0.9)
    discounted = zeros(Float32, length(rewards), 1)
    discounted[end] = rewards[end]

    for i=(length(rewards)-1):-1:1
        discounted[i] = rewards[i] + γ * discounted[i+1]
    end
    return discounted
end

function train!(w, opts, env, o)
    totalr = [0.0]
    g = lossgradient(w, env, o; totalr=totalr)
    update!(w, g, opts)
    return totalr[1]
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="(c) Ozan Arkan Can, 2018. Demonstration of the REINFORCE algorithm on the discrete action space."
    @add_arg_table s begin
        ("--env_id"; default="CartPole-v1"; help="environment name")
        ("--episodes"; arg_type=Int; default=20; help="number of episodes")
        ("--gamma"; arg_type=Float64; default=0.99; help="doscount factor")
        ("--threshold"; arg_type=Int; default=1000; help="stop the episode even it is not terminal after number of steps exceeds the threshold")
        ("--lr"; arg_type=Float64; default=0.01; help="learning rate")
        ("--render"; help = "render the environment"; action = :store_true)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
    end

    srand(12345)
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    
    o = parse_args(args, s)
    o["atype"] = eval(parse(o["atype"]))

    env = GymEnv(o["env_id"])
    seed!(env, 12345)

    INPUT = env.observation_space.shape[1]
    OUTPUT = env.action_space.n

    w = init_params(INPUT, OUTPUT, o["atype"])
    opts = Dict()
    for k in keys(w)
        opts[k] = Rmsprop(lr=o["lr"])
    end

    for i=1:o["episodes"]
        total = train!(w, opts, env, o)
        println("episode $i , total rewards: $total")
    end
end

PROGRAM_FILE=="reinforce_discrete.jl" && main(ARGS)

end
