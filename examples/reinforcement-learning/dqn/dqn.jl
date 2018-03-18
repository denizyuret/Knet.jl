try
    Pkg.installed("Gym")
catch
    Pkg.clone("https://github.com/ozanarkancan/Gym.jl")
    ENV["GYM_ENVS"] = "atari:algorithmic:box2d:classic_control"
    Pkg.build("Gym")
end

for p in ("ArgParse", "Knet", "JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
julia dqn.jl

This example implements the DQN algorithm from 
`Playing atari with deep reinforcement learning.`
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). 
arXiv preprint arXiv:1312.5602.
"""
module DQN

using Gym, ArgParse, Knet, JLD

include("replay_buffer.jl")
include("mlp.jl")
include("piecewise_schedule.jl")

function loss(w, states, actions, targets; nh=1)
    qvals = predict_q(w, states; nh=1)
    nrows = size(qvals, 1)
    index = actions + nrows*(0:(length(actions)-1))
    qpred = reshape(qvals[index], size(targets)...)
    mse = sum(abs2, targets-qpred) / size(states, 2)
    return mse
end

lossgradient = gradloss(loss)

function train!(w, prms, states, actions, targets; nh=1)
    g, mse = lossgradient(w, states, actions, targets; nh=nh)
    update!(w, g, prms)
    return mse
end

function dqn_learn(w, opts, env, buffer, exploration, o)
    total = 0.0
    readytosave = 10000
    episode_rewards = Float32[]
    frames = Float32[]
    ob_t = reset!(env)
    
    for fnum=1:o["frames"]
        o["render"] && render(env)
        #process the raw ob
        ob_t_reshaped = reshape(ob_t, size(ob_t)..., 1)
        if !o["play"] && rand() < value(exploration, fnum)
            a = sample(env.action_space)
        else
            obses_t = encode_recent(buffer, ob_t_reshaped; stack=o["stack"])
            inp = convert(o["atype"], obses_t)
            qvals = predict_q(w, inp; nh=length(o["hiddens"]))
            a = indmax(Array(qvals)) - 1
        end
        
        ob_t, reward, done, _ = step!(env, a)
        total += reward 

        if !o["play"]
            #process the raw ob
            ob_tp1_reshaped = reshape(ob_t, size(ob_t)..., 1)
            push!(buffer, ob_t_reshaped, a+1, reward, ob_tp1_reshaped, done)
            
            if can_sample(buffer, o["bs"])
                obses_t, actions, rewards, obses_tp1, dones = sample_batch(buffer, o["bs"]; stack=o["stack"])
                obses_tp1 = convert(o["atype"], obses_tp1)
                nextq = predict_q(w, obses_tp1; nh=length(o["hiddens"]))
                nextq = Array(nextq)
                maxs = maximum(nextq,1)
                nextmax = sum(nextq .* (nextq.==maxs), 1)
                nextmax = reshape(nextmax, 1, length(nextmax))
                targets = reshape(rewards,1,length(rewards)) .+ (o["gamma"] .* nextmax .* dones)
                obses_t = convert(o["atype"], obses_t)
                targets = convert(o["atype"], targets)
                mse = train!(w, opts, obses_t, actions, targets; nh=length(o["hiddens"]))
            end

            if o["save"] != "" && fnum > readytosave
                save_model(w, o["save"])
                readytosave += 10000
            end
        end

        if done
            ob_t = reset!(env)
            o["printinfo"] && println("Frame: $fnum , Total reward: $total, Exploration Rate: $(value(exploration, fnum))")
            push!(episode_rewards, total)
            push!(frames, fnum)
            total = 0.0
        end
    end
    return episode_rewards, frames
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "(c) Ozan Arkan Can, 2018. An implementation of the deep q network."
    @add_arg_table s begin
        ("--frames"; arg_type=Int; default=100; help="number of frames")
        ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        ("--gamma"; arg_type=Float64; default=0.99; help="discount factor")
        ("--hiddens"; arg_type=Int; nargs='+'; default=[32]; help="number of units in the hiddens for the mlp")
        ("--env_id"; default="CartPole-v0")
        ("--render"; action=:store_true)
        ("--memory"; arg_type=Int; default=1000; help="memory size")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--stack"; arg_type=Int; default=4; help="length of the frame history")
        ("--save"; default=""; help="model name")
        ("--load"; default=""; help="model name")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--play"; action=:store_true; help="only play")
        ("--printinfo"; action=:store_true; help="print the training messages")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s)
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o["atype"] = eval(parse(o["atype"]))
    srand(12345)
    env = GymEnv(o["env_id"])
    seed!(env, 12345)

    INPUT = env.observation_space.shape[1] * o["stack"]
    OUTPUT = env.action_space.n

    if o["load"] == ""
        w = init_weights(INPUT, o["hiddens"], OUTPUT, o["atype"]) 
    else
        w = load_model(o["load"], o["atype"])
    end
    
    opts = Dict()
    for k in keys(w)
        opts[k] = Rmsprop(lr=o["lr"])
    end

    buffer = ReplayBuffer(o["memory"])

    exploration = PiecewiseSchedule([(0, 1.0),
                                     (round(Int, o["frames"]/5), 0.1),
                                     (round(Int, o["frames"]/3.5), 0.1)])

    rewards, frames = dqn_learn(w, opts, env, buffer, exploration, o)
end

PROGRAM_FILE == "dqn.jl" && main(ARGS)

end
