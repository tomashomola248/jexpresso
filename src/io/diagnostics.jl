function mass_conservation(q_St)

    npoin = length(q_St.qn[:,1])

    mass_init = sum(q_St.qe[:,1])
    mass = abs(sum(q_St.qn[:,1]) - mass_init)/mass_init

    @printf "      mass loss = %.4e\n" mass
    
end
