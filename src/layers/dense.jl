struct ARPDense{M, B, F, I}
    W::M
    b::B
    σ::F

    L::I
    xmin_lim::I
    xmax_lim::I
    xq_scalar::I
    auto_rot::Bool
    eps::Float64

    function ARPDense(W::M, L::I, xmin_lim::I, xmax_lim::I;
                        bias=true, σ::F=identity, xq_scalar="auto", auto_rot=true, eps=1e-5) where {M<:AbstractMatrix, I<:Integer, F}
        b = Flux.create_bias(W, bias, size(W, 1))
        xq_scalar = xq_scalar == "auto" ? 2*xmax_lim - xmin_lim : xq_scalar
        if L === nothing
            @assert !auto_rot
        end
        new{M, typeof(b), F, I}(W, b, σ, L, xmin_lim, xmax_lim, xq_scalar, auto_rot, eps)
    end
end

function ARPDense((in, out)::Pair{<:Integer, <:Integer}, L::Integer, xmin_lim::Integer, xmax_lim::Integer;
                    bias=true, σ=identity, init=Flux.glorot_uniform, xq_scalar="auto", auto_rot=true, eps=1e-5)
    ARPDense(init(out, in), L, xmin_lim, xmax_lim; bias, σ, xq_scalar, auto_rot, eps)
end

Flux.@functor ARPDense
function (m::ARPDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(m.σ, x)

    f(x) = σ.(m.W * x' .+ m.b)
    outputs = f(x)

    if m.auto_rot
        inputs_xq = m.xq_scalar * ones(size(x))
        fOfxQ = f(inputs_xq)
        den = abs.(fOfxQ) .+ m.eps
        ρ = m.L ./ den
        outputs = σ.(ρ .* outputs)
    end

    return outputs
end

function Base.show(io::IO, m::ARPDense)
  print(io, "ARPDense(", size(m.W, 2), " => ", size(m.W, 1))
  m.σ == identity || print(io, ", ", m.σ)
  m.b == false && print(io, "; bias=false")
  print(io, ")")
end
