include("../AbstractProblems.jl")
include("../../kernel/mesh/mesh.jl")
include("../../io/plotting/jeplots.jl")

function initialize(SD::NSD_1D, ET::AdvDiff, mesh::St_mesh, inputs::Dict, OUTPUT_DIR::String, TFloat)

    @info " Initialize fields for AdvDiff ........................ "
    
    qinit = Array{TFloat}(undef, mesh.npoin, 1)
    neqs  = 1
    q     = define_q(SD, mesh.nelem, mesh.npoin, mesh.ngl, neqs)
    
    σ = Float64(64.0)
    for iel_g = 1:mesh.nelem
        for i=1:mesh.ngl
            
            ip = mesh.connijk[i,iel_g]
            x  = mesh.x[ip]
            
            q.qn[ip, 1] = exp(-σ*x*x)
            u           = 0.8
                   
        end
    end
    #Exact solution
    q.qe = copy(q.qn)

   # q.mass_init = sum(q.qe[:,1])
    
    #------------------------------------------
    # Plot initial condition:
    # Notice that I scatter the points to
    # avoid sorting the x and q which would be
    # becessary for a smooth curve plot.
    #------------------------------------------
    title = string( "Tracer: initial condition")
    plot_curve(mesh.x, q.qn[:,1], title, string(OUTPUT_DIR, "/INIT.png"))
    
    @info " Initialize fields for AdvDiff ........................ DONE"
    
    return q
end


function initialize(SD::NSD_2D, ET::AdvDiff, mesh::St_mesh, inputs::Dict, OUTPUT_DIR::String, TFloat)

    @info " Initialize fields for AdvDiff ........................ "
        
    ngl  = mesh.nop + 1
    nsd  = mesh.nsd
    neqs = 1    
    q    = define_q(SD, mesh.nelem, mesh.npoin, mesh.ngl, neqs)
    
    test_case = "kopriva.5.3.5"
    if (test_case == "kopriva.5.3.5")
        #Cone properties:
        ν = inputs[:νx] 
        if ν == 0.0
            ν = 0.01
        end
        σ = 1.0/ν
        (xc, yc) = (-0.5, -0.5)
        u = 0.8
        v = 0.8
        for iel_g = 1:mesh.nelem
            for i=1:ngl
                for j=1:ngl

                    ip = mesh.connijk[i,j,iel_g]
                    x  = mesh.x[ip]
                    y  = mesh.y[ip]

                    q.qn[ip,1] = exp(-σ*((x - xc)*(x - xc) + (y - yc)*(y - yc)))

                end
            end
        end

        #qexact at t=tend
        t = inputs[:tend]
        σ = 1.0/(ν*(4.0t + 1.0))
        for ip = 1:mesh.npoin
            q.qe[ip] = 1.0/(4t + 1.0) * exp(-σ*((mesh.x[ip] - u*t - xc)^2 + (mesh.y[ip] - v*t - yc)^2))
        end
    end
    
    #------------------------------------------
    # Plot initial condition:
    # Notice that I scatter the points to
    # avoid sorting the x and q which would be
    # becessary for a smooth curve plot.
    #------------------------------------------
    title = string( "Tracer: initial condition")
    jcontour(SD, mesh.x, mesh.y, q.qn[:,1], title, string(OUTPUT_DIR, "/INIT.png"))
    
    @info " Initialize fields for AdvDiff ........................ DONE"
    
    return q
end
