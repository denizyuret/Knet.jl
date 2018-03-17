if Pkg.installed("Gym") == nothing
    Pkg.clone("https://github.com/ozanarkancan/Gym.jl")
    ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
    Pkb.build("Gym")
end

for p in ("ArgParse", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
julia actor_critic_discrete.jl

This example implements the online version of Actor-Critic algorithm. 
It is a variant of REINFORCE algorithm. The TD-error is 
used as the advantage function.
"""
module ACTOR_CRITIC

using Gym, ArgParse, Knet, AutoGrad

function predict(w, x, name; nh=1)
    inp = x
    for i=1:nh
        inp = relu.(w[name*"w_$i"] * inp .+ w[name*"b_$i"])
    end
    out = w[name*"w_out"] * inp .+ w[name*"b_out"]
    return out
end

function sample_action(linear)
    linear = Array(linear)
    probs = exp.(linear) ./ sum(exp.(linear), 1)
    c_probs = cumsum(probs)
    return indmax(c_probs .> rand())
end

@zerograd sample_action(linear)

function actor(w, ob; nh=1)
    linear = predict(w, ob, "actor";nh=nh)
    action = sample_action(linear)
    return action, linear
end

critic(w, ob;nh=1) = predict(w, ob, "critic"; nh=nh)

function loss(w, ob, t, env, o; output_of_step=nothing)
    action, linear = actor(w, ob; nh=length(o["actor"]))
    v_s = critic(w, ob; nh=length(o["critic"]))

    ob, reward, done, _ = step!(env, action-1)
    ob = convert(o["atype"], ob)
    push!(output_of_step, (ob, reward, done))
    
    v_sp1 = done ? Float32(0.0) : critic(w, ob; nh=length(o["critic"]))

    #reward shaping
    #=
    if done && t <= 195
        reward = Float32(-1.0)
    else
        reward = Float32(reward)
    end
    =#
    reward = Float32(reward)

    δ = reward + o["gamma"] * v_sp1 - v_s

    critic_loss = sum(δ .* δ)# size of δ is (1,1)
    actor_loss = sum(-logp(linear, 1)[action] .*  AutoGrad.getval(δ)) / size(linear, 2)
    
    return actor_loss + critic_loss
end

lossgradient = grad(loss)

function play_episode!(w, opts, env, o)
    ob = reset!(env)
    total = 0
    ob = convert(o["atype"], ob)
    for t=1:env.spec.max_episode_steps
        output = Any[]
        g = lossgradient(w, ob, t, env, o; output_of_step=output)
        update!(w, g, opts)

        #converted next state, reward and isTerminal
        ob, reward, done = output[1]

        total += reward
        o["render"] && render(env)

        done && break
    end
    return total
end

function init_weights(name, input, hiddens, output, atype)
    w = Dict()
    inp = input
    for i=1:length(hiddens)
        w[name*"w_$i"] = 0.01*randn(hiddens[i], inp)
        w[name*"b_$i"] = zeros(hiddens[i])
        inp = hiddens[i]
    end

    w[name*"w_out"] = 0.01*randn(output, hiddens[end])
    w[name*"b_out"] = zeros(output, 1)

    for k in keys(w)
        w[k] = convert(atype, w[k])
    end

    return w
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="(c) Ozan Arkan Can, 2018. Demonstration of the online Actor-Critic algorithm."
    @add_arg_table s begin
        ("--env_id"; default="CartPole-v0"; help="environment name")
        ("--actor"; nargs='+'; arg_type=Int; default=[100]; help="number of hiddens for the actor")
        ("--critic"; nargs='+'; arg_type=Int; default=[100]; help="number of hiddens for the critic")
        ("--episodes"; arg_type=Int; default=20; help="number of episodes")
        ("--gamma"; arg_type=Float32; default=Float32(0.99); help="doscount factor")
        ("--lr"; arg_type=Float32; default=Float32(0.01); help="learning rate")
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

    w = init_weights("actor", INPUT, o["actor"], OUTPUT, o["atype"])
    merge!(w, init_weights("critic", INPUT, o["critic"], 1, o["atype"]))
    opts = Dict()
    for k in keys(w)
        opts[k] = Adam(lr=o["lr"])
    end

    for i=1:o["episodes"]
        total = play_episode!(w, opts, env, o)
        println("episode $i , total rewards: $total")
    end
end

PROGRAM_FILE=="actor_critic_discrete.jl" && main(ARGS)

end
