using ARP
using Flux
using Test

@testset "ARPDense" begin
    model = Chain(
        Dense(123=>20),
        ARPDense(20=>10, 4, 0, 1, σ=sigmoid)
    )
    params = Flux.params(model)
    dense_params_size = size(params[1])
    arp_params_size = size(params[3])

    @test dense_params_size == (20, 123)
    @test arp_params_size == (10, 20)
end
