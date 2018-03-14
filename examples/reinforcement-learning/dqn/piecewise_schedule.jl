"""
    https://github.com/berkeleydeeprlcourse/homework/blob/60b3ef08c2dca3961efb63b32683bb588571f226/sp17_hw/hw3/dqn_utils.py#L49
"""

linear_interpolation(l, r, alpha) = l + alpha * (r - l)

struct PiecewiseSchedule
    endpoints
    interpolation
end

PiecewiseSchedule(endpoints) = PiecewiseSchedule(endpoints, linear_interpolation)

function value(schedule::PiecewiseSchedule, t)
    for ((l_t, l),(r_t,r)) in zip(schedule.endpoints[1:end-1],schedule.endpoints[2:end])
        if l_t <= t && t < r_t
            alpha = (t - l_t) / (r_t - l_t)
            return schedule.interpolation(l, r, alpha)
        end
    end
    return schedule.endpoints[end][2]
end
