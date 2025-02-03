module conicUtils

using ..utils

export interpolationConic
export add
export sub
export center
export discriminant
export evaluate
export quadraticFormula
export to_string
export intersectWithStandardFormLine

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
    return conicEquation( (u2-u1)^2, 2*(u2-u1)*(u3-u1), (u3-u1)^2, 2*u1*(u2-u1), 2*u1*(u3-u1), u1^2 )
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

# intersect the conic with the line ax + by + c = 0
function intersectWithStandardFormLine(eq::conicEquation, a::Float64, b::Float64, c::Float64)
    if b != 0.0
        qa = a/b
        qc = c/b

        # find x which is roots of quadratic equation formed by substituting y = (-ax-c)/b
        intersections_x = quadraticFormula( eq.A - qa*eq.B + (qa^2)*eq.C, -qc*eq.B + 2*qa*qc*eq.C + eq.D - qa*eq.E, (qc^2)*eq.C - qc*eq.E + eq.F )
        if intersections_x[1] == Inf
            return ((Inf,Inf),(Inf,Inf))
        else
            x1 = intersections_x[1]
            y1 = (-a*x1-c)/b

            if intersections_x[2] == Inf
                return ((x1,y1),(Inf,Inf))
            else
                x2 = intersections_x[2]
                y2 = (-a*x2-c)/b
                return ((x1,y1),(x2,y2))
            end
        end
    else
        qc = -c/a
        intersections_y = quadraticFormula( eq.C, eq.B*qc + eq.E, eq.A*qc*qc + eq.D*qc + eq.F )
        if intersections_y[1] == Inf
            return ((Inf,Inf),(Inf,Inf))
        elseif intersections_y[2] == Inf
            return ((qc,intersections_y[1]),(Inf,Inf))
        else
            return ((qc,intersections_y[1]),(qc,intersections_y[2]))
        end
    end
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

function to_string(eq::conicEquation)
    return string(eq.A) * "x^2 + " * string(eq.B) * "xy + " * string(eq.C) * "y^2 + " * string(eq.D) * "x + " * string(eq.E) * "y + " * string(eq.F)
end

end