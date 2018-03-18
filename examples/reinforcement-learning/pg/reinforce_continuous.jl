try
    Pkg.installed("Gym")
catch
    Pkg.clone("https://github.com/ozanarkancan/Gym.jl")
    ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
    Pkg.build("Gym")
end

for p in ("ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
julia reinforce_continous.jl

This example implements the REINFORCE algorithm from
`Simple statistical gradient-following algorithms for
connectionist reinforcement learning.`,  Williams, Ronald J.
Machine learning, 8(3-4):229–256, 1992. This example also
demonstrates the usage of the `@zerograd` function for
stopping the gradient flow.
"""
module REINFORCE_CONTINUOUS

using Gym, ArgParse, Knet, AutoGrad

function lrelu(x, leak=0.2)
    f1 = Float32(0.5 * (1 + leak))
    f2 = Float32(0.5 * (1 - leak))
    return f1 * x + f2 * abs.(x)
end

function predict_μ(w, ob)
    hidden = lrelu.(w["w1"] * ob .+ w["b1"])
    linear = w["w2"] * hidden .+ w["b2"]
    return linear
end

function logpdf(μ, x; σ=Float32(1.0))    
    fac = Float32(-log(sqrt(2pi)))
    r = (x-μ) / σ
    return -r.* r.* Float32(0.5) - Float32(log(σ)) + fac
end

function sample_action(μ; σ=1.0)
    μ = convert(Array{Float32}, μ)
    a = μ .+ randn(size(μ)) .* σ
end

@zerograd sample_action(mu)

function play(w, ob)
    μ = predict_μ(w, ob)
    action = sample_action(μ)
    return action, μ
end

function play_episode(w, env, o)
    ob = reset!(env)
    rewards = Float32[]
    μs = Any[]
    actions = Array{Float32,1}()
    total = 0

    for t=1:env.spec.max_episode_steps
        ob_inp = convert(o["atype"], reshape(ob, size(ob, 1), 1))
        action, μ = play(w, ob_inp)
        push!(μs, μ)
        ob, reward, done, _ = step!(env, action-1)

        total += reward[1]

        push!(rewards, reward[1])
        push!(actions, action[1])

        o["render"] && render(env)

        done && break
    end
    return μs, actions, rewards, total
end

function loss(w, env, o; totalr=nothing)
    μs, actions, rewards, total = play_episode(w, env, o)
    totalr[1] = total

    actions = convert(o["atype"], reshape(actions, size(w["w2"],1), length(actions)))
    discounted = discount(rewards; γ=o["gamma"])
    discounted = discounted .- mean(discounted)#mean R as a baseline
    discounted = discounted ./ (std(discounted) + 1e-6)

    discounted = convert(o["atype"], reshape(discounted, 1, size(actions, 2)))
    -sum((logpdf(hcat(μs...), actions) .* discounted)) / size(μs, 2)
end

lossgradient = grad(loss)

function init_params(input, hidden, output, atype)
    w = Dict()

    w["w1"] = xavier(hidden, input)
    w["b1"] = zeros(hidden, 1)
    w["w2"] = xavier(output, hidden)
    w["b2"] = zeros(output, 1)

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
    s.description="(c) Ozan Arkan Can, 2018. Demonstration of the REINFORCE algorithm on the continuous action space."
    @add_arg_table s begin
        ("--env_id"; default="Pendulum-v0"; help="environment name")
        ("--episodes"; arg_type=Int; default=20; help="number of episodes")
        ("--gamma"; arg_type=Float64; default=0.9; help="doscount factor")
        ("--threshold"; arg_type=Int; default=1000; help="stop the episode even it is not terminal after number of steps exceeds the threshold")
        ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        ("--render"; help = "render the environment"; action = :store_true)
        ("--hidden"; arg_type=Int; default=64; help="hidden units")
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

    o = parse_args(args, s)
    o["atype"] = eval(parse(o["atype"]))

    env = GymEnv(o["env_id"])
    seed!(env, 12345)

    INPUT = env.observation_space.shape[1]
    OUTPUT = env.action_space.shape[1]

    w = init_params(INPUT, o["hidden"], OUTPUT, o["atype"])
    opts = Dict()
    for k in keys(w)
        opts[k] = Rmsprop(lr=o["lr"])
    end

    for i=1:o["episodes"]
        total = train!(w, opts, env, o)
        println("episode $i , total rewards: $total")
    end
end

PROGRAM_FILE=="reinforce_continuous.jl" && main(ARGS)

end
