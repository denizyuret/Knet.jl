ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
for p in ("Gym","ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module REINFORCE_CONTINOUS

using Gym, ArgParse, Knet, AutoGrad

function lrelu(x, leak=0.2)
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x+ f2 *abs(x)
end

function predict_linear(w, ob)
    hidden = lrelu.(w["w1"] * ob .+ w["b1"])
    linear = w["w2"] * hidden .+ w["b2"]
    return linear
end

function logpdf(μ, x; sigma=1.0)    
    fac = -log(sqrt(2pi))
    r = (x-μ) / sigma
    return -r.* r.* 0.5 - log(sigma) + fac
end

function sample_action(mu; sigma=1.0)
    a = mu + randn() * sigma
end

@zerograd sample_action(mu)

function play(w, ob)
    linear = predict_linear(w, ob)
    action = sample_action(Array(linear))
    return action, linear
end

function play_episode(w, env, o)
    ob = reset!(env)
    rewards = Float32[]
    linears = Any[]
    actions = Array{Float32,1}()
    total = 0

    for t=1:env.spec.max_episode_steps
        ob_inp = convert(o["atype"], reshape(ob, size(ob, 1), 1))
        action, linear = play(w, ob_inp)
        push!(linears, linear)
        ob, reward, done, _ = step!(env, action-1)

        total += reward[1]

        push!(rewards, reward[1])
        push!(actions, action[1])

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

function main(ARGS)
    s = ArgParseSettings()
    s.description="(c) Ozan Arkan Can, 2018. REINFORCE Algorithm."
    s.exc_handler=ArgParse.debug_handler
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

    o = parse_args(s)
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

main(ARGS)
end
