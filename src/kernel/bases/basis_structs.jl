#
# This file contains all the structs definitions
# S. Marras, Feb 2022
#
include("../abstractTypes.jl")

export St_Lagrange
export St_Legendre
export St_lgl
export St_lg
export St_cg
export build_lgl!
export LegendreGaussLobattoNodesAndWeights


abstract type AbstractIntegrationPointAndWeights end
abstract type AbstractInterpolationBasis end
abstract type AbstractSpaceDimensions end

struct LagrangeBasis <: AbstractInterpolationBasis end

mutable struct St_Legendre{TFloat}
    
    """
    struct St_Legendrea{TFloat<:Real}
          legendre  :: TFloat
          dlegendre :: TFloat
           q        :: TFloat -> q  = legendre(p+1)  - legendre(p-1)
           dq       :: TFloat -> dq = dlegendre(p+1) - dlegendre(p-1)
    end
    """
    
    legendre  :: TFloat
    dlegendre :: TFloat
    q         :: TFloat
    dq        :: TFloat
end

mutable struct St_Chebyshev{TFloat} <:AbstractIntegrationPointAndWeights
    chebyshev ::TFloat
end

mutable struct St_lgl{TFloat} <:AbstractIntegrationPointAndWeights
    ξ::Array{TFloat}
    ω::Array{TFloat}
end

mutable struct St_lg{TFloat} <:AbstractIntegrationPointAndWeights
    ξ::Array{TFloat}
    ω::Array{TFloat}
end

mutable struct St_cg{TFloat} <:AbstractIntegrationPointAndWeights
    ξ::Array{TFloat}
    ω::Array{TFloat}
end

mutable struct St_cgl{TFloat} <:AbstractIntegrationPointAndWeights
    ξ::Array{TFloat}
    ω::Array{TFloat}
end

mutable struct St_Lagrange{TFloat} <:AbstractInterpolationBasis
    ψ::Matrix{TFloat}
    dψ::Matrix{TFloat}
end

function basis_structs_ξ_ω!(::AbstractPointsType, nop::TInt) end

function basis_structs_ξ_ω!(ξωtype::LG, nop::TInt)
    
    lg = St_lg{TFloat}(zeros(TFloat, nop+1),
                       zeros(TFloat, nop+1))
    
    build_Integration_points!(lg, nop)

    return lg
end

function basis_structs_ξ_ω!(ξωtype::LGL, nop::TInt)
    
    lgl = St_lgl{TFloat}(zeros(TFloat, nop+1),
                         zeros(TFloat, nop+1))
    
    build_Integration_points!(lgl, nop)

    return lgl
end

function basis_structs_ξ_ω!(ξωtype::CG, nop::TInt)
    
    cg = St_cg{TFloat}(zeros(TFloat, nop+1),
                       zeros(TFloat, nop+1))
    
    build_Integration_points!(cg, nop)

    return cg
end


function basis_structs_ξ_ω!(ξωtype::CGL, nop::TInt)
    
    cgl = St_cgl{TFloat}(zeros(TFloat, nop+1),
                         zeros(TFloat, nop+1))
    
    build_Integration_points!(cgl, nop)

    return cgl
end


function build_Integration_points!(::AbstractIntegrationPointAndWeights,nop::TInt) end
 
function build_Integration_points!(cg::St_cg, nop::TInt)
  build_cg!(cg,nop)
end

function build_Integration_points!(cgl::St_cgl,nop::TInt)
  build_cgl!(cgl,nop)
end

function build_Integration_points!(lg::St_lg,nop::TInt)
  Legendre = St_Legendre{TFloat}(0.0,0.0,0.0,0.0)
  build_lg!(Legendre,lg,nop)
end

function build_Integration_points!(lgl::St_lgl,nop::TInt)
  Legendre = St_Legendre{TFloat}(0.0,0.0,0.0,0.0)
  build_lgl!(Legendre,lgl,nop)
end

function build_Interpolation_basis!(TP::LagrangeBasis, ξ, ξq, T::Type{Float64})

    Nξ = size(ξ,1)  - 1
    Qξ = size(ξq,1) - 1

    N  = (Nξ + 1)
    Q  = (Qξ + 1)
    
    basis = St_Lagrange{T}(zeros(N,Q), zeros(N,Q))
    (basis.ψ, basis.dψ) = LagrangeInterpolatingPolynomials_classic(ξ, ξq, T)
    
    return basis
end


function build_lgl!(Legendre::St_Legendre, lgl::St_lgl, nop::TInt)

    """
      Evaluate recursion, the Legendre polynomial of order p
      and its Derivatives at coordinate x
     
      L_{p}  --> legendre of order p
      L'_{p} --> dlegendr of order p
      
      and
     
      q  = L_{p+1}  -  L_{p-1}
      q' = L'_{p+1} -  L'_{p-1} 
      
      Algorithm 22+24 of Kopriva's book
      
    """
    size::Int8 = nop+1
    
    lgl.ξ = zeros(Float64, size)
    lgl.ω = zeros(Float64, size)
    
    #LGL nodes
    LegendreGaussLobattoNodesAndWeights!(Legendre, lgl, nop);
    
    return lgl;
end


function build_cg!(cg::St_cg, nop::TInt)
    
    println(" # Compute Chebyshev-Gauss nodes ........................ ")
    size::Int8=nop+1
    cg.ξ = zeros(Float64, size)
    cg.ω = zeros(Float64, size)

    #CG nodes
    ChebyshevGaussNodesAndWeights!(cg,nop)
    
    for j=1:size
        println( " # ξ cheby, ω =: ", " ", cg.ξ[j], " " , cg.ω[j])
    end

    println(" # Compute Chebyshev-Gauss nodes ........................ DONE")
    return cg
end

function build_cgl!(cgl::St_cgl, nop::TInt)
    
    println(" # Compute Chebyshev-Gauss-Lobatto nodes ........................ ")
    
    size::Int8=nop+1
    cgl.ξ = zeros(Float64, size)
    cgl.ω = zeros(Float64, size)

    #CGL nodes
    ChebyshevGaussLobattoNodesAndWeights!(cgl,nop)
    
    for j=1:size        
        println( " # ξ cheby, ω =: ", " ", cgl.ξ[j], " " , cgl.ω[j])
    end
    
    println(" # Compute Chebyshev-Gauss-Lobatto nodes ........................ DONE")
    return cgl
end

function build_lg!(Legendre::St_Legendre,lg::St_lg,nop)
    size::Int8=nop+1
    lg.ξ = zeros(Float64, size)
    lg.ω = zeros(Float64, size)

    #LG nodes
    LegendreGaussNodesAndWeights!(Legendre,lg,nop)
    return lg
end

function ChebyshevGaussNodesAndWeights!(cg::St_cg, nop::TInt)
    """
         Compute the Nodes for the Chebyshev-Gauss Quadrature
         using Algorithm 26 of Kopriva's book 
      """
    for j=0:nop
        cg.ξ[j+1]=-cospi((2*j+1)/(2*nop+2))
        cg.ω[j+1]=π/(nop+1)
    end
end

function ChebyshevGaussLobattoNodesAndWeights!(cgl::St_cgl,nop::TInt)
    """
         Compute the Nodes for the Chebyshev-Gauss-Lobatto Quadrature
         using Algorithm 26 of Kopriva's book 
      """
    for j=0:nop
        cgl.ξ[j+1]=-cospi(j/nop)
        cgl.ω[j+1]=π/nop
    end
    cgl.ω[1]=cgl.ω[1]/2
    cgl.ω[nop+1]=cgl.ω[nop+1]/2
end


function LegendreGaussNodesAndWeights!(Legendre::St_Legendre, lg::St_lg, nop::TInt)
    """
          Compute the Nodes and Weights for the Legendre-Gauss Quadrature
          using Algorithm 23 of Kopriva's book valid for nop ≤  200
    """

    println( " # Compute LG nodes ........................")
    NITER = 100
    TOL = 4*eps(Float64)
    Δ::Float64=0.0
    if (nop == 0)
        lg.ξ[1]=0
        lg.ω[1]=2
    elseif (nop == 1)
        lg.ξ[1] = -sqrt(1/3)
        lg.ω[1] = 1
        lg.ξ[2] = - lg.ξ[1]
        lg.ω[2] = lg.ω[1]
    else
        for j=0:floor(Int,(nop+1)/2)-1
            lg.ξ[j+1] = - cospi((2*j+1)/(2*nop+2))
            for k=0:NITER
                LegendreAndDerivativeAndQ!(Legendre, nop+1, lg.ξ[j+1])
                Δ = -Legendre.legendre/Legendre.dlegendre
                lg.ξ[j+1]=lg.ξ[j+1]+Δ
                if (abs(Δ)≤TOL)
                    break
                end
            end
            LegendreAndDerivativeAndQ!(Legendre, nop+1, lg.ξ[j+1])
            lg.ξ[nop-j+1] = - lg.ξ[j+1]
            lg.ω[j+1]=2/(1-lg.ξ[j+1]^2)/(Legendre.dlegendre)^2
            lg.ω[nop-j+1]=lg.ω[j+1]
        end
    end
    if (mod(nop,2) == 0)
        LegendreAndDerivativeAndQ!(Legendre, nop+1, 0.0)
        lg.ξ[TInt(nop/2)+1] = 0
        lg.ω[TInt(nop/2)+1] = 2/Legendre.dlegendre^2
    end

    
    for j=1:nop+1       
        println( " # ξ, ω =: ", " ", lg.ξ[j], " " , lg.ω[j])
    end
    
    println(" # Compute LG nodes ........................ DONE")
    
end

function LegendreGaussLobattoNodesAndWeights!(Legendre::St_Legendre, lgl::St_lgl, nop::TInt)
     """
          Compute the Nodes and Weights for the Legendre-Gauss-Lobatto Quadrature
     """

    NITER = 100
    TOL = 4*eps()
    
    ξ0 ::Float64=0.0
    ξ1 ::Float64=1.0
    ξP ::Float64=1.0
    ξj ::Float64=0.0
    ξj2::Float64=0.0
    L2 ::Float64=0.0
    
    ω0 ::Float64=0.0
    ω1 ::Float64=0.0
    ωP ::Float64=0.0
    
    Δ  ::Float64=0.0
    
    println( " # Compute LGL nodes ........................")
    
    for j=1:nop+1
	lgl.ξ[j] = 0.0;
	lgl.ω[j] = 1.0;
    end
    
    if (nop == 1)
	ξ0           = -1.0
	ω0           =  1.0
	ξ1           =  1.0
	ω1           =   ω0
        
	lgl.ξ[1]     =   ξ0
	lgl.ξ[nop+1] =   ξ1
	lgl.ω[1]     =   ω0
	lgl.ω[nop+1] =   ω1
    else 
	ξ0            = -1.0
	ω0            =  TFloat(2.0/(nop*(nop + 1)))
	ξP            =  1.0
	ωP            =   ω0
        
	lgl.ξ[1]      =   ξ0
	lgl.ξ[nop+1]  =   ξP
	lgl.ω[1]      =   ω0
	lgl.ω[nop+1]  =   ωP
	
        for jj = 2:floor(Int,(nop + 1)/2) 
	    j = jj - 1
	    ξj = -cos((j + 0.25)*π/nop - 3.0/(8.0*nop*π*(j + 0.25)))
	    lgl.ξ[jj] = ξj;
            
            for k = 0:NITER
	        LegendreAndDerivativeAndQ!(Legendre, nop, ξj)
	        Δ = -Legendre.q/Legendre.dq
	        ξj    =  ξj + Δ
	        
	        if (abs(Δ) <= TOL*abs(ξj))
                    break
                end
	    end
            LegendreAndDerivativeAndQ!(Legendre, nop, ξj)
	    lgl.ξ[jj]      =  ξj
	    lgl.ξ[nop+1-j] = -ξj
	    xj2            =  ξj*ξj
	    L2             = Legendre.legendre*Legendre.legendre
	    lgl.ω[jj]      = 2.0/(nop*(nop + 1.0)*L2)
	    lgl.ω[nop+1-j] = lgl.ω[jj]
            
        end
    end
    
    if (mod(nop,2) == 0)
	LegendreAndDerivativeAndQ!(Legendre, nop, 0.0);
	lgl.ξ[TInt(nop/2)+1] = 0.0;
	
	L2           = Legendre.legendre*Legendre.legendre;
	lgl.ω[TInt(nop/2)+1] = 2.0/(nop*(nop + 1.0)*L2);
    end

    for j=1:nop+1       
        println( " # ξ, ω =: ", " ", lgl.ξ[j], " " , lgl.ω[j])
    end
    
    println(" # Compute LGL nodes ........................ DONE")
    
end

function LegendreAndDerivativeAndQ!(Legendre::St_Legendre, nop::TInt, x::TFloat)
    
    """
             Evaluate by recursion, the Legendre polynomial of order p
             its Derivatives at coordinate x, and q:
     
              L_{p}  --> legendre of order p
              L'_{p} --> dlegendr of order p
              q  = L_{p+1}  -  L_{p-1}
              q' = L'_{p+1} -  L'_{p-1} 
     
              Algorithm 24 of Kopriva's book
     
              Simone Marras, October 2021

              
              Note that this algorithm looses precision at high enough nop
              if we intendt to use particularly large nop we should examine Yakimiw 1996 (Yassine) 

         """
    TFloat=Float64
    
    a   ::TFloat=0.0
    b   ::TFloat=0.0
    ϕ   ::TFloat=0.0
    dϕ  ::TFloat=0.0
    ϕp1 ::TFloat=0.0
    dϕp1::TFloat=0.0
    ϕm1 ::TFloat=0.0
    dϕm1::TFloat=0.0
    ϕm2 ::TFloat=0.0
    dϕm2::TFloat=0.0
    
    
    #st_legendre Legendre;  
    if (nop == 0) #Order 0 case
        ϕ    = 1.0
        dϕ   = 0.0
        q    = x
        dq   = 1.0
    elseif (nop == 1)
        ϕ    = x
        dϕ   = 1.0
        q    = 0.5*(3*x^2 - 2) - 1
        dq   = 3*x
    else
        ϕm2  = 1.0
	ϕm1  = x
	dϕm2 = 0.0
	dϕm1 = 1.0

        #Construct Nth Order Legendre Polynomial
	for k=2:nop
             
            ϕ    = x*ϕm1*(2.0*k - 1.0)/k - ϕm2*(k - 1.0)/k
	    dϕ   = dϕm2 + (2.0*k - 1.0)*ϕm1
	    
	    ϕp1  = x*ϕ*(2.0*k + 1.0)/(k+1) - ϕm1*k/(k+1)
	    dϕp1 = dϕm1 + (2.0*k - 1.0)*ϕ
            
            q  =  ϕp1 -  ϕm1
            dq = dϕp1 - dϕm1
            
	    ϕm2 = ϕm1
	    ϕm1 = ϕ

	    dϕm2 = dϕm1
	    dϕm1 = dϕ
            
	end
    end
    
    Legendre.legendre  =  ϕ
    Legendre.dlegendre = dϕ
    Legendre.q         =  q
    Legendre.dq        = dq
    
end

function ChebyshevPolynomial!(Chebyshev::St_Chebyshev,nop::TInt,x::TFloat,Ks::TInt)
    """
          Evaluate by recursion the Chebyshev Polynomial
          and switch to direct evaluation when nop > Ks
          uses Algorithm 21 of Kopriva's book, 
          NOTE Ks depends should never exceed 70 regardless of machine architecture
          Yassine Tissaoui March 2022
       """
    T2::TFloat=0.0
    T1::TFloat=0.0
    T::TFloat=0.0
    if (nop == 0) #case for order 0
        Chebyshev.chebyshev = 1
    elseif (nop ==1) #case for order 1
        Chebyshev.chebyshev = x
    elseif (nop ≤ Ks)
        T2=1
        T1=x
        for j=2:nop
            T=2*x*T1 - T2
            T2=T1
            T1=T
        end
        Chebyshev.chebyshev = T
    else
        T=cos(nop*acos(x))
        Chebyshev.chebyshev = T
    end
end

function LagrangeInterpolatingPolynomials_classic(ξ, ξq, TFloat)
"""
    LagrangeInterpolatingPolynomials_classic(ξ, ξq, N, Q, TFloat)
    ξ::set of N interpolation points (e.g. LGL points)
    ξq::point to interpolate to (e.g. quadrature points of points within the element)
    
    Algorithm 3.1 + 3.2 from Giraldo's book

    from https://github.com/fxgiraldo/Element-based-Galerkin-Methods/blob/master/Projects/Project_01_1D_Interpolation/For_Instructors/julia/lagrange_basis.jl

"""
    N = size(ξ,1) - 1
    Q = size(ξq,1) - 1
    
    #Initialize arrays
    L    = zeros(TFloat, N+1, Q+1)
    dLdx = zeros(TFloat, N+1, Q+1)
 
    for l=1:Q+1
        xl = ξq[l]

        #Construct Basis
        for i=1:N+1
            
            xi        = ξ[i]
            L[i,l]    = 1.0
            dLdx[i,l] = 0.0
            for j=1:N+1
                xj = ξ[j]

                #L
                if (j != i)
                    L[i,l] = L[i,l]*(xl - xj)/(xi - xj)
                end
                
                ddL=1
                if (j != i)
                    for k=1:N+1
                        xk = ξ[k]
                        
                        #dL/dx
                        if (k !=i && k !=j)
                            ddL = ddL*(xl - xk)/(xi - xk)
                        end
                    end
                    dLdx[i, l] = dLdx[i, l] + ddL/(xi - xj)
                end
            end
        end
    end

    return (L, dLdx)
end

function function_space_wrapper(inputs::Dict, )

    Nξ = inputs[:nop]
    lexact_integration = inputs[:lexact_integration]    
    PT    = inputs[:problem]
    
    #--------------------------------------------------------
    # Create/read mesh
    # return mesh::St_mesh
    # and Build interpolation nodes
    #             the user decides among LGL, GL, etc. 
    # Return:
    # ξ = ND.ξ.ξ
    # ω = ND.ξ.ω
    #--------------------------------------------------------
    mesh = mod_mesh_mesh_driver(inputs)
    
    #--------------------------------------------------------
    # Build interpolation and quadrature points/weights
    #--------------------------------------------------------
    ξω  = basis_structs_ξ_ω!(inputs[:interpolation_nodes], mesh.nop)    
    ξ,ω = ξω.ξ, ξω.ω
    if lexact_integration
        #
        # Exact quadrature:
        # Quadrature order (Q = N+1) ≠ polynomial order (N)
        #
        QT  = Exact() #Quadrature Type
        QT_String = "Exact"
        Qξ  = Nξ + 1
        
        ξωQ   = basis_structs_ξ_ω!(inputs[:quadrature_nodes], mesh.nop)
        ξq, ω = ξωQ.ξ, ξωQ.ω
    else  
        #
        # Inexact quadrature:
        # Quadrature and interpolation orders coincide (Q = N)
        #
        QT  = Inexact() #Quadrature Type
        QT_String = "Inexact"
        Qξ  = Nξ
        ξωq = ξω
        ξq  = ξ
        ω   = ξω.ω
    end
    if (mesh.nsd == 1)
        SD = NSD_1D()
    elseif (mesh.nsd == 2)
        SD = NSD_2D()
    elseif (mesh.nsd == 3)
        SD = NSD_3D()
    else
        error(" Drivers.jl: Number of space dimnnsions unknow! CHECK Your grid!")
    end
    
    return (; basis, ω, metrics, mesh, SD, QT)
end
