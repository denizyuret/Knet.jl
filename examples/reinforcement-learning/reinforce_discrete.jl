ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
for p in ("Gym","ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Gym, ArgParse, Knet

function predict_linear(w, ob)
    linear = w["w"] * ob .+ w["b"]
    return linear
end

function loss(w, observations, actions, discounted_rewards)
    linear = predict_linear(w, observations)
    logps = sum(logp(linear, 1) .* actions, 1)
    -sum(logps .* discounted_rewards) ./ size(observations, 2)
end

lossgradient = grad(loss)

function sample_action(linear)
    probs = exp.(linear) ./ sum(exp.(linear), 1)
    c_probs = cumsum(probs)
    return indmax(c_probs .> rand())
end

function play(w, ob)
    linear = Array(predict_linear(w, ob))
    action = sample_action(linear)
    return action
end

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

function train!(w, opts, observations, actions, rewards, atype; γ=0.9)
    y = zeros(size(w["w"], 1), length(actions))

    for i=1:length(actions)
        y[actions[i], i] = 1.0
    end

    actions = convert(atype, y)

    discounted = discount(rewards; γ=γ)
    discounted = discounted .- mean(discounted)#mean R as a baseline

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
        ("--env_id"; default="CartPole-v1"; help="environment name")
        ("--episodes"; arg_type=Int; default=20; help="number of episodes")
        ("--gamma"; arg_type=Float64; default=0.99; help="doscount factor")
        ("--threshold"; arg_type=Int; default=1000; help="stop the episode even it is not terminal after number of steps exceeds the threshold")
        ("--lr"; arg_type=Float64; default=0.01; help="learning rate")
        ("--render"; help = "render the environment"; action = :store_true)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
    end

    srand(12345)

    o = parse_args(s)
    o["atype"] = eval(parse(o["atype"]))

    env = GymEnv(o["env_id"])

    INPUT = env.observation_space.shape[1]
    OUTPUT = env.action_space.n

    w = init_params(INPUT, OUTPUT, o["atype"])
    opts = Dict()
    for k in keys(w)
        opts[k] = Rmsprop(lr=o["lr"])
    end

    for i=1:o["episodes"]
        ob = reset!(env)
        rewards = Float32[]
        observations = Any[]
        actions = Array{Int,1}()
        total = 0

        for t=1:env.spec.max_episode_steps
            ob_inp = convert(o["atype"], reshape(ob, size(ob, 1), 1))
            push!(observations, ob_inp)
            action = play(w, ob_inp)
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

        train!(w, opts, observations, actions, rewards, o["atype"]; γ=o["gamma"])

        println("episode $i , total rewards: $total")
    end
end

main(ARGS)
