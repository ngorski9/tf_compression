module conicUtils

using ..utils

export checkDSEllipse
export checkRSEllipse
export checkDSEllipseProper
export checkRSEllipseProper

# This entire file assumes the interpolation paradigm where we are linearly interpolating three tensors in a triangle.
# We assume that the first one is (0,0), the second (1,0), and the third (0,1). However, the goal here is just to determine
# whether one of the ellipses that we need is actually inside of the cell, so the locations of the vertices do not actually matter.

struct conicEquation
    A::Float64
    B::Float64
    C::Float64
    D::Float64
    E::Float64
    F::Float64
end

function interpolationConic(u1::Float64, u2::Float64, u3::Float64)
    return conicEquation( u2-u1, 2*(u2-u1)*(u3-u1), (u3-u1)^2, 2*u1*(u2-u1), 2*u1*(u3-u1), u1^2 )
end

function add(eq1::conicEquation, eq2::conicEquation)
    return conicEquation( eq1.A+eq2.A, eq1.B+eq2.B, eq1.C+eq2.C, eq1.D+eq2.D, eq1.E+eq2.E, eq1.F+eq2.F )
end

function sub(eq1::conicEquation, eq2::conicEquation)
    return conicEquation( eq1.A-eq2.A, eq1.B-eq2.B, eq1.C-eq2.C, eq1.D-eq2.D, eq1.E-eq2.E, eq1.F-eq2.F )
end

function center(eq::conicEquation)
    return ( (eq.B*eq.E - 2*eq.C*eq.D)/(4*eq.A*eq.C-eq.B^2), (eq.B*eq.D - 2*eq.A*eq.E)/(4*eq.A*eq.C-eq.B^2) )
end

function discriminant(eq::conicEquation)
    return eq.B^2 - 4*eq.A*eq.C
end

function evaluate(eq::conicEquation, x::Float64, y::Float64)
    return eq.A*x^2 + eq.B*x*y + eq.C*y^2 + eq.D*x + eq.E*y + eq.F
end

function quadraticFormula(a::Float64, b::Float64, c::Float64)
    disc = b^2-4*a*c

    if disc < 0.0
        return (Inf,Inf)
    end

    sr = sqrt(disc)

    if a == 0.0
        if b == 0.0
            return (Inf, Inf)
        else
            return (-c/b,Inf)
        end
    elseif a > 0.0
        return ( (-b - sr) / (2*a), (-b + sr) / (2*a) )
    else
        return ( (-b + sr) / (2*a), (-b - sr) / (2*a) )
    end

end

# This may preserve a few too many cells, but I believe it is very rare anyway.
function ellipseInCell(eq::conicEquation)
    # check that the equation represents an ellipse
    if discriminant(eq) >= 0
        return false
    end

    # check that the center lies within the cell
    c = center(eq)
    if !(0 <= c[1] <= 1 && 0 <= c[2] <= 1 && c[1] <= c[2])
        return false
    end

    # find the sign of the center to determine the sign of evaluation that is the inside of the ellipse.
    # That way we can test if the vertices are inside of the ellipse.
    s = sign(evaluate(eq, c...))
    return sign(evaluate(eq, 0.0, 0.0)) != s && sign(evaluate(eq, 1.0, 0.0)) != s && sign(evaluate(eq, 0.0, 1.0)) != s
end

# verify that the ellipse is properly contained within the given cell (i.e. it really does not intersect the boundary)
function ellipseInCellProper(eq::conicEquation)
    if ellipseInCell(eq)

        roots1 = quadraticFormula(eq.A, eq.D, eq.F) # edge from (0,0) to (1,0) parametrized by (t,0)
        roots2 = quadraticFormula(eq.C, eq.E, eq.F) # edge from (0,0) to (0,1) parametrized by (0,t)
        roots3 = quadraticFormula(eq.A-eq.B+eq.C, eq.B-2*eq.C+eq.D-eq.E, eq.C+eq.E+eq.F) # edge from (0,1) to (1,0) parametrized by (t,1-t)

        return !(0 <= roots1[1] <= 1 || 0 <= roots1[2] <= 1 || 0 <= roots2[1] <= 1 || 0 <= roots2[2] <= 1 || 0 <= roots3[1] <= 1 || 0 <= roots3[2] <= 1)
    else
        return false
    end
end

function checkDSEllipse(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
    dConic = interpolationConic( tensor1[1,1]+tensor1[2,2], tensor2[1,1]+tensor2[2,2], tensor3[1,1]+tensor3[2,2] )
    sConic1 = interpolationConic( tensor1[1,1]-tensor1[2,2], tensor2[1,1]-tensor2[2,2], tensor3[1,1]-tensor3[2,2] )
    sConic2 = interpolationConic( tensor1[2,1]+tensor1[1,2], tensor2[2,1]+tensor2[1,2], tensor3[2,1]+tensor3[1,2] )

    conic = sub(add(sConic1, sConic2), dConic)

    return ellipseInCell(conic)
end

function checkRSEllipse(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
    rConic = interpolationConic( tensor1[2,1]-tensor1[1,2], tensor2[2,1]-tensor2[1,2], tensor3[2,1]-tensor3[1,2] )
    sConic1 = interpolationConic( tensor1[1,1]-tensor1[2,2], tensor2[1,1]-tensor2[2,2], tensor3[1,1]-tensor3[2,2] )
    sConic2 = interpolationConic( tensor1[2,1]+tensor1[1,2], tensor2[2,1]+tensor2[1,2], tensor3[2,1]+tensor3[1,2] )

    conic = sub(add(sConic1, sConic2), rConic)

    return ellipseInCell(conic)
end

# Do not just approximate, but actually show that the ellipse really is inside the cell.
function checkDSEllipseProper(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
    dConic = interpolationConic( tensor1[1,1]+tensor1[2,2], tensor2[1,1]+tensor2[2,2], tensor3[1,1]+tensor3[2,2] )
    sConic1 = interpolationConic( tensor1[1,1]-tensor1[2,2], tensor2[1,1]-tensor2[2,2], tensor3[1,1]-tensor3[2,2] )
    sConic2 = interpolationConic( tensor1[2,1]+tensor1[1,2], tensor2[2,1]+tensor2[1,2], tensor3[2,1]+tensor3[1,2] )

    conic = sub(add(sConic1, sConic2), dConic)

    return ellipseInCellProper(conic)
end

# Do not just approximate, but actually show that the ellipse really is inside the cell.
function checkRSEllipseProper(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
    rConic = interpolationConic( tensor1[2,1]-tensor1[1,2], tensor2[2,1]-tensor2[1,2], tensor3[2,1]-tensor3[1,2] )
    sConic1 = interpolationConic( tensor1[1,1]-tensor1[2,2], tensor2[1,1]-tensor2[2,2], tensor3[1,1]-tensor3[2,2] )
    sConic2 = interpolationConic( tensor1[2,1]+tensor1[1,2], tensor2[2,1]+tensor2[1,2], tensor3[2,1]+tensor3[1,2] )

    conic = sub(add(sConic1, sConic2), rConic)

    return ellipseInCellProper(conic)
end

end