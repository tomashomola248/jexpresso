using QuadGK

# Define problem parameters
N = 20   # number of elements
a = -1.0 # left endpoint of domain
b = 1.0  # right endpoint of domain
h = (b - a) / N   # element size

# Define element function
function element(x)
    # Define basis functions
    phi1(x) = 0.5*(1 - x)
    phi2(x) = 0.5*(1 + x)
    # Define element matrix
    E = [1/h -1/h; -1/h 1/h]
    # Compute element vector
    b = zeros(2)
    b[1] = quadgk(x -> phi1(x)*x, -1, 1)[1] / h
    b[2] = quadgk(x -> phi2(x)*x, -1, 1)[1] / h
    # Compute element solution
    u = E \ b
    return u
end

# Populate elements and construct connectivity matrix
C = zeros(2, N)
U = zeros(2, N)
for i = 1:N
    C[1, i] = (i-1)*h + a
    C[2, i] = i*h + a
    U[:, i] = element([-1, 1])
end
return C