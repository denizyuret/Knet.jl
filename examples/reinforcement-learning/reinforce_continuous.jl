ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
for p in ("Gym","ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Gym, ArgParse, Knet

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

function logpdf(mu, x; sigma=1.0)    
    fac = -log(sqrt(2pi))
    r = (x-mu)/ sigma
    return -r.* r.* 0.5 - log(sigma) + fac
end

function loss(w, observations, actions, discounted_rewards)
    mus = predict_linear(w, observations)
    -sum((logpdf(mus, actions) .* discounted_rewards)) / size(mus, 2)
end

lossgradient = grad(loss)

function sample_action(mu; sigma=1.0)
    a = mu + randn() * sigma
end

function play(w, ob)
    linear = Array(predict_linear(w, ob))
    action = sample_action(linear)
    return action
end

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

function train!(w, opts, observations, actions, rewards, atype; γ=0.9)
    actions = convert(atype, reshape(actions, size(w["w2"],1), length(actions)))

    discounted = discount(rewards; γ=γ)
    discounted = discounted .- mean(discounted)#standardize the rewards to be unit normal
    discounted = discounted ./ (std(discounted) + 1e-6)

    discounted = convert(atype, reshape(discounted, 1, size(actions, 2)))
    observations = hcat(observations...)
    g = lossgradient(w, observations, actions, discounted)
    update!(w, g, opts)
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
        ob = reset!(env)
        rewards = Float32[]
        observations = Any[]
        means = Float32[]
        total = 0

        for t=1:env.spec.max_episode_steps
            ob_inp = convert(o["atype"], reshape(ob, size(ob, 1), 1))
            push!(observations, ob_inp)
            action = play(w, ob_inp)
            ob, reward, done, _ = step!(env, action)

            total += reward[1]

            push!(rewards, reward[1])
            push!(means, action[1])

            if o["render"]
                render(env)
            end

            if done
                break
            end
        end

        train!(w, opts, observations, means, rewards, o["atype"]; γ=o["gamma"])

        println("episode $i , total rewards: $total")
    end
end

main(ARGS)
