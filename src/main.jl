cd(dirname(@__FILE__))
# This Julia script simulates the Vicsek model, using the link cell method.
# First we provide some general purpose functions for linear regression, plot, save and load data.
# Then we define the main functions for the Vicsek model, including the initialization, the link cell method, the main simulation, and the order parameter analysis.
# Finally, we provide the calculations for replicating the result of the original Vicsek paper.

# Importing Packages
using Random, StatsBase, LaTeXStrings, Plots, ProgressMeter, Serialization, DataFrames, GLM
# Random: For generating random numbers.
# StatsBase: Provides statistical support functions.
# LaTeXStrings and Plots: For creating and managing plots with LaTeX labels.
# ProgressMeter: To display progress meters for long-running operations.
# Serialization: For saving and loading data.
# DataFrames: To handle, manipulate, and analyze data in tabular form.
# GLM: Generalized linear models for statistical analysis.

default(fontfamily="Computer Modern", linewidth=0.85, markersize = 3, grid=false, xminorticks=10, yminorticks=10, dpi = 250) # plots aesthetics

struct Parameters
    L::Float64 # Size of the box
    N::Int64 # Number of birds
    v0::Float64 # Velocity of birds
    η::Float64 # Noise amplitude
    ηs::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64} # Noise amplitudes for order parameter analysis
    R::Float64 # Interaction radius
    t_steps::Int64 # Number of steps for the simulation
    Δt::Float64 # Time step
    therm_steps::Int64 # Number of thermalization steps
    measure_steps::Int64 # Number of measurements for each η
    MCS::Int64 # Number actualization between measurements
end

mutable struct State
    x::Array{Float64, 1} # Position in x
    y::Array{Float64, 1} # Position in y
    vx::Array{Float64, 1} # Velocity in x
    vy::Array{Float64, 1} # Velocity in y
    θ::Array{Float64, 1} # Angle
end

mutable struct LinkCellMethod
    Nc::Int64 # Number of cells
    rc::Float64 # Length of the cell
    cells_neighbours::Array{Set{Tuple{Int,Int}}, 2} # List of neighbour cells for each cell
    birds_cells::Array{Set{Int}, 2} # Matrix of list of birds inside i,j cell
end

mutable struct OrderParameter
    φη::Array{Float64, 1}  # Polar order parameter for a η
    φs::Array{Float64, 1}  # Polar order parameter for each η
    δφs::Array{Float64, 1}  # Polar order parameter for each η
end

# -------------------------------------------------------
# General purpose functions: linear regression, plot, save and load data
# -------------------------------------------------------
"""
    linear_regression(X, Y, verbose=false)

Perform linear regression on the given data.

Parameters:
- `X`: Independent variable data.
- `Y`: Dependent variable data.
- `verbose`: A boolean indicating whether to print the regression results. Default is `false`.

Returns:
A dictionary containing the regression results including intercept, slope, their errors, and R-squared value.
"""
function linear_regression(X, Y, verbose = false)
    # Linear regression without errors
    df = DataFrame(x = X, y = Y)
    # Perform linear regression
    linear_model = lm(@formula(y ~ x), df)

    # Extract coefficients and standard errors
    B, A = coef(linear_model)
    σ_B, σ_A = stderror(linear_model)

    # Calculate R-squared
    r_squared = r2(linear_model)

    # Return results as a dictionary
    results = Dict(
        "Intercept" => B,
        "Slope" => A,
        "Intercept_Error" => σ_B,
        "Slope_Error" => σ_A,
        "R_squared" => r_squared
    )
    verbose ? println("y = ($A pm $σ_A) x + ($B pm $σ_B); r^2 = $r_squared") : nothing

    return results
end

"""
    plot_lr(X::Array, Y::Array, results::Dict, xlabel="x", ylabel="y", scatter_label="Data Points", fit_label="Linear Regression Fit", save=false, filename="Linear_regression.png")

Plot linear regression results.

Parameters:
- `X::Array`: Independent variable data.
- `Y::Array`: Dependent variable data.
- `results::Dict`: Dictionary containing the regression results including slope, intercept, their errors, and R-squared value.
- `xlabel::String`: Label for the x-axis. Default is "x".
- `ylabel::String`: Label for the y-axis. Default is "y".
- `scatter_label::String`: Label for the data points in the scatter plot. Default is "Data Points".
- `fit_label::String`: Label for the linear regression fit line. Default is "Linear Regression Fit".
- `save::Bool`: A boolean indicating whether to save the plot as an image. Default is `false`.
- `filename::String`: Name of the file to save the plot. Default is "Linear_regression.png".

Returns:
The plot displaying the data points and the linear regression fit line.
"""
function plot_lr(X::Array,Y::Array, results::Dict, xlabel="x", ylabel="y", scatter_label::String = "Data Points", fit_label::String = "Linear Regression Fit", save = false, filename::String = "Linear_regression.png")
    # Extract results
    A, B = results["Slope"], results["Intercept"]
    σ_A, σ_B = results["Slope_Error"], results["Intercept_Error"]
    r2 = results["R_squared"]

    # Plot results
    x_fit = range(start = minimum(X), stop = maximum(X), length = 1000)
    y_fit = A .* x_fit .+ B

    plot(x_fit, y_fit, label = fit_label)
    scatter!(X, Y, label = scatter_label, xlabel=LaTeXString(xlabel), ylabel=LaTeXString(ylabel))

    # Save figure
    save == true ? savefig(filename) : nothing

    # Display the plot
    display(current())
end

"""
    save_data(filename, data)

Save data to a file using serialization.

Parameters:
- `filename::String`: Name of the file to save the data.
- `data`: Data to be saved.

Returns:
Nothing. Saves the data to the specified file.
"""
function save_data(filename, data)
    open(filename, "w") do io
        serialize(io, data)
    end
end

"""
    load_data(filename)

Load data from a file using deserialization.

Parameters:
- `filename::String`: Name of the file to load the data from.

Returns:
The deserialized data from the specified file.
"""
function load_data(filename)
    open(filename, "r") do io
        return deserialize(io)
    end
end

#----------------------------------------------------------------
# Main Functions: Vicsek Model 
#----------------------------------------------------------------
"""
    initial_condition(params::Parameters, state::State)

Generate initial conditions for a system.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system.

Returns:
Nothing. Updates the state with initial conditions.
"""
function initial_condition(params::Parameters, state::State)
    state.x = params.L * rand(Float64, params.N)
    state.y = params.L * rand(Float64, params.N)
    state.θ = 2π * rand(Float64, params.N)
    state.vx = params.v0 * cos.(state.θ)
    state.vy = params.v0 * sin.(state.θ)
end

v(params::Parameters, state::State) = params.v0 .* [cos.(state.θ), sin.(state.θ)] # Velocity


"""
    which_cell(state::State, lcell::LinkCellMethod, bird::Int64)

Determine which cell a particle belongs to in the link cell method.

Parameters:
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `bird::Int64`: Index of the particle.

Returns:
Indices (i, j) of the cell in which the particle resides.
"""
function which_cell(state::State, lcell::LinkCellMethod, bird::Int64)
    i = floor(Int64, state.x[bird]/lcell.rc) + 1 
    j = floor(Int64, state.y[bird]/lcell.rc) + 1
    # +1 since in julia, we start indexing at 1
    return i, j
end

"""
    birds_to_cells(params::Parameters, state::State, lcell::LinkCellMethod)

Assign particles to cells in the link cell method.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.

Returns:
Nothing. Updates the `birds_cells` field of the link cell method object.
"""
function birds_to_cells(params::Parameters, state::State, lcell::LinkCellMethod)
    for i in 1:lcell.Nc, j in 1:lcell.Nc
        xmin = (i-1)*lcell.rc; xmax = i*lcell.rc
        ymin = (j-1)*lcell.rc; ymax = j*lcell.rc
        in_x = xmin .≤ state.x .≤ xmax
        in_y = ymin .≤ state.y .≤ ymax
        bird_idx = findall(in_x .& in_y)
        lcell.birds_cells[i, j] = Set(bird_idx)
    end
end

"""
    intialize_cells(params::Parameters, state::State, lcell::LinkCellMethod)

Initialize cells and assign particles to cells in the link cell method.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.

Returns:
Nothing. Updates the `Nc`, `rc`, `cells_neighbours`, and `birds_cells` fields of the link cell method object.
"""
function intialize_cells(params::Parameters, state::State, lcell::LinkCellMethod)
    lcell.Nc = floor(Int64, params.L/params.R) 
    lcell.rc = params.L/lcell.Nc 

    # Generate neighbour cells for each cell
    for i in 1:lcell.Nc, j in 1:lcell.Nc
        lcell.cells_neighbours[i,j] = Set([(i, j), (i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j+1), (i-1, j-1), (i+1, j-1), (i-1, j+1)])
        # Apply periodic boundary conditions (If L+1 set to 1, and 0 to L)
        lcell.cells_neighbours[i, j] = Set(mod1.(tup, lcell.Nc) for tup in lcell.cells_neighbours[i,j])

        # Initialize birds in cells
        lcell.birds_cells[i, j] = Set{Int}()
    end

    # Initialize birds in cells
    birds_to_cells(params, state, lcell)
end

"""
    neighbours_birds_idxs(params::Parameters, state::State, lcell::LinkCellMethod, bird::Int64)

Find indices of neighboring particles within a certain radius around a given particle using link cell method.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `bird::Int64`: Index of the particle.

Returns:
Indices of neighboring particles within the specified radius around the given particle.
"""
function neighbours_birds_idxs(params::Parameters, state::State, lcell::LinkCellMethod, bird::Int64)
    i, j = which_cell(state, lcell, bird)
    neigh_cells = lcell.cells_neighbours[i, j] # Set of neighbour cells
    neigh_birds = Set{Int}() # Initialize an empty set for neighboring birds

    for cell in neigh_cells
        for neigh_b in lcell.birds_cells[cell...]
            Δx = abs(state.x[bird] - state.x[neigh_b]); Δx_bc = min(Δx, params.L - Δx)
            Δy = abs(state.y[bird] - state.y[neigh_b]); Δy_bc = min(Δy, params.L - Δy)
            r = hypot(Δx_bc, Δy_bc)
            if r <= params.R
                push!(neigh_birds, neigh_b)
            end
        end
    end

    return neigh_birds
end

"""
    new_step!(params::Parameters, state::State, lcell::LinkCellMethod, η::Float64=params.η)

Perform a new time step in the simulation.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions and velocities.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `η::Float64`: Noise parameter. Default is taken from `params.η`.

Returns:
Nothing. Updates the positions, angles, and velocities of particles in the system, as well as the cells they belong to.
"""
function new_step!(params::Parameters, state::State, lcell::LinkCellMethod, η::Float64 = params.η)
    # Update positions with periodic boundary conditions
    state.x .+= state.vx * params.Δt; state.x = mod.(state.x, params.L)
    state.y .+= state.vy * params.Δt; state.y = mod.(state.y, params.L)

    # Compute mean angle
    sinθ = sin.(state.θ); cosθ = cos.(state.θ)
    θ_avg = similar(state.θ)
    for bird in 1:params.N
        neigh_birds = neighbours_birds_idxs(params, state, lcell, bird)
        # Since we then divide, we don't take the avg
        sx, sy = 0.0, 0.0
        for nb in neigh_birds
            sx += cosθ[nb]
            sy += sinθ[nb]
        end
        # sx = sum([cosθ[nb] for nb in neigh_birds])
        # sy = sum([sinθ[nb] for nb in neigh_birds])
        θ_avg[bird] = atan(sy, sx)
    end

    # Update angles and velocities
    ξ = rand(params.N) .- 0.5
    state.θ .= θ_avg .+ η * ξ
    state.vx, state.vy = v(params, state)
    
    # Update cells
    birds_to_cells(params, state, lcell) 
end

"""
    simulation(params::Parameters, state::State, lcell::LinkCellMethod, steps::Int64=params.t_steps, η::Float64=params.η)

Run the simulation for a specified number of steps.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: Initial state of the system.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `steps::Int64`: Number of steps to run the simulation. Default is taken from `params.t_steps`.
- `η::Float64`: Noise parameter. Default is taken from `params.η`.

Returns:
Nothing. Updates the state of the system by running the simulation for the specified number of steps.
"""
function simulation(params::Parameters, state::State, lcell::LinkCellMethod, steps::Int64 = params.t_steps, η::Float64 = params.η)
    for _ in 1:steps
        new_step!(params, state, lcell, η)
    end
end

"""
    calc_φ(params::Parameters, state::State)

Calculate the order parameter φ of the system.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle velocities.

Returns:
The order parameter φ of the system.
"""
function calc_φ(params::Parameters, state::State)
    vx_avg = sum(state.vx); vy_avg = sum(state.vy)
    v_mod = hypot(vx_avg, vy_avg)
    return v_mod/(params.N * params.v0)
end

"""
    ηVicsek(params::Parameters, state::State, lcell::LinkCellMethod, ord_param::OrderParameter, η::Float64=params.η)

Run Vicsek model simulations with varying noise parameter η.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: Initial state of the system.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `ord_param::OrderParameter`: Object to store order parameter values.
- `η::Float64`: Noise parameter. Default is taken from `params.η`.

Returns:
Nothing. Updates the order parameter object with φ values calculated for each noise level.
"""
function ηVicsek(params::Parameters, state::State, lcell::LinkCellMethod, ord_param::OrderParameter, η::Float64 = params.η)
    # Calculate φ for each velocity configuration given by the Monte Carlo steps
    for i in 1:params.measure_steps
        ord_param.φη[i] = calc_φ(params, state)

        # Update state
        simulation(params, state, lcell, params.MCS, η)
    end
end

"""
    v_vs_η(params::Parameters, state::State, lcell::LinkCellMethod, ord_param::OrderParameter, verbose::Bool=false)

Calculate the order parameter φ for different noise levels (η) in the Vicsek model.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: Initial state of the system.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `ord_param::OrderParameter`: Object to store order parameter values.
- `verbose::Bool`: A boolean indicating whether to print verbose output. Default is `false`.

Returns:
A tuple containing arrays of order parameter values (φs) and their standard deviations (δφs).
"""
function v_vs_η(params::Parameters, state::State, lcell::LinkCellMethod, ord_param::OrderParameter, verbose::Bool = false)
    # Calculation for different noise amplitudes
    if verbose
        println("---------- VICSEK MODEL ----------")
        println("N = $(params.N), L = $(params.L), therm_steps = $(params.therm_steps), measure_steps = $(params.measure_steps), MCS = $(params.MCS)")
    end

    progress = Progress(length(params.ηs), 1, "Fig 2 for $(params.N)", 50)
    # Calculate for each temperature
    for (i, η) in enumerate(params.ηs)

        # Thermalization. We do therm_steps MCS
        simulation(params, state, lcell, params.therm_steps, η)

        # Measure φ for measure_steps taking MCS in between
        ηVicsek(params, state, lcell, ord_param, η)

        # Save data
        ord_param.φs[i] = mean(ord_param.φη)
        ord_param.δφs[i] = std(ord_param.φη)

        verbose ? println("Birds configuration for η = $η is done!") : nothing
        next!(progress)
    end
    return ord_param.φs, ord_param.δφs
end

#----------------------------------------------------------------------
# Calculations
#----------------------------------------------------------------------

# Gifs. Fig 1 in gif format
# -------------------------------------------------- 
"""
    plot_flock(state::State, params::Parameters, t::Int64)

Plot the flocking behavior of particles at a specific time step.

Parameters:
- `state::State`: State of the system containing particle positions and velocities.
- `params::Parameters`: Parameters of the system.
- `t::Int64`: Time step to plot.

Returns:
A plot showing the flocking behavior of particles at the specified time step.
"""
function plot_flock(state::State, params::Parameters, t::Int64)
    p = plot(state.x, state.y, seriestype = :scatter, markersize = 2, color = :black, xlims = (0, params.L), ylims = (0, params.L), legend = false)
    quiver!(state.x, state.y, quiver = 7 .* (state.vx, state.vy), arrow = true)
    xlabel!(L"x")
    ylabel!(L"y")
    title!(L"t = %$t")
    return p
end

"""
    produce_gif(L::Float64, η::Float64, N::Int64, t_steps::Int64, filename::String="test.gif")

Produce a GIF animation showing the flocking behavior of particles over time.

Parameters:
- `L::Float64`: Length of the simulation domain.
- `η::Float64`: Noise parameter.
- `N::Int64`: Number of particles.
- `t_steps::Int64`: Number of time steps to simulate.
- `filename::String`: Name of the output GIF file. Default is "test.gif".

Returns:
Nothing. Saves the GIF animation showing the flocking behavior.
"""
function produce_gif(L::Float64, η::Float64, N::Int64, t_steps::Int64, filename::String = "test.gif")
    # Common parameters
    v0 = 0.03; R = 1; Δt = 1

    # Initialize Parameters
    params = Parameters(L, N, v0, η, 5:-0.1:0, R, t_steps, Δt, 1000, 500, 5) 

    # Initialize State
    x = Array{Float64}(undef, N)
    y = Array{Float64}(undef, N)
    vx = Array{Float64}(undef, N)
    vy = Array{Float64}(undef, N)
    θ = Array{Float64}(undef, N)
    state = State(x, y, vx, vy, θ)
    initial_condition(params, state)

    # Initialize Cells
    Nc = floor(Int64, L/R)
    cells_neighbours = Array{Set{Tuple{Int,Int}}, 2}(undef, Nc, Nc)
    birds_cells = Array{Set{Int}, 2}(undef, Nc, Nc)
    lcell = LinkCellMethod(Nc, L/Nc, cells_neighbours, birds_cells)
    intialize_cells(params, state, lcell)

    # Producing GIF
    progress = Progress(t_steps, 1, "Computing frames... ", 50)

    anim = @animate for t in 0:t_steps
        plot_flock(state, params, t)
        new_step!(params, state, lcell)
        next!(progress)
    end

    @time gif(anim, filename, fps = 15)
end

N = 40; t_steps = 200 # Test
L = 10.0; η = 0.1; filename = "animation_test.gif" # Test
# N = 300; t_steps = 1500
# L = 7.0; η = 2.0; filename = "animation_1a.gif" # Plot 1a
# L = 25.0; η = 0.1; filename = "animation_1b.gif" # Plot 1b
# L = 5.0; η = 0.1; filename = "animation_1c.gif" # Plot 1c
produce_gif(L, η, N, t_steps, filename)


# Figures 2 & 3 
# -------------------------------------------------- 
"""
    fig2_and_X(Ns::Array{Int64, 1}, ρ::Float64, save=false)

Generate figures showing the order parameter and its fluctuations for different particle numbers.

Parameters:
- `Ns::Array{Int64, 1}`: Array of particle numbers.
- `ρ::Float64`: Density of the particles.
- `save::Bool`: A boolean indicating whether to save the generated figures. Default is `false`.

Returns:
Nothing. Displays and optionally saves the generated figures.
"""
function fig2_and_X(Ns::Array{Int64, 1}, ρ::Float64, save = false)
    plt2 = plot(xlabel = L"\eta", ylabel = L"v_a")
    pltX = plot(xlabel = L"\eta", ylabel = L"\sigma[v_a]")

    # Common parameters
    v0 = 0.03; η = 0.1; ηs = 5:-0.1:0; R = 1; t_steps = 500; Δt = 1; therm_steps = 1000; measure_steps = 500; MCS = 5
     
    for N in Ns
        L = sqrt(N/ρ)
        params = Parameters(L, N, v0, η, ηs, R, t_steps, Δt, therm_steps, measure_steps, MCS) # Initialize Parameters

        # Initialize State
        x = Array{Float64}(undef, params.N)
        y = Array{Float64}(undef, params.N)
        vx = Array{Float64}(undef, params.N)
        vy = Array{Float64}(undef, params.N)
        θ = Array{Float64}(undef, params.N)

        state = State(x, y, vx, vy, θ) 
        initial_condition(params, state)

        # Initialize Cells
        Nc = floor(Int64, params.L/params.R); rc = NaN
        cells_neighbours = Array{Set{Tuple{Int,Int}}, 2}(undef, Nc, Nc)
        birds_cells = Array{Set{Int}, 2}(undef, Nc, Nc)
        lcell = LinkCellMethod(Nc, rc, cells_neighbours, birds_cells) # LinkCellMethod
        intialize_cells(params, state, lcell)

        # Order Parameter
        φη = Array{Float64}(undef, params.measure_steps)
        φs = Array{Float64}(undef, length(params.ηs))
        δφs = Array{Float64}(undef, length(params.ηs))
        ord_param = OrderParameter(φη, φs, δφs)

        # Check if file exists
        filename = "v_as_N$(N).dat"
        # Calculations or loading data if the file exists.
        if isfile(filename)
            println("Loading data for N = $N... ")
            φs, δφs = load_data(filename)
        else
            println("Calculating data for N = $N... ")
            @time φs, δφs = v_vs_η(params, state, lcell, ord_param) #, true)
            save_data(filename, (φs, δφs))
        end

        # Plot
        scatter!(plt2, ηs, φs, label = L"$N$ = %$N")
        plot!(pltX, ηs, δφs, label = L"$N$ = %$N")

        δφs_max = maximum(δφs); idx = findall(δφs .== δφs_max)
        ηc = ηs[idx][1]
        println("N = $N, ηc = $(ηc)")
    end
    # Display the plot
    display(plt2)
    display(pltX)

    # Save fig
    save ? savefig(plt2, "fig2.png") : nothing
    save ? savefig(pltX, "Fluctuations.png") : nothing
    
end

# @time fig2_and_X([40, 100, 400, 4000, 10000], 4.0)

"""
    fig3(Ns::Array{Int64, 1}, ρ::Float64, ηcs::Array{Float64, 1}, save=false)

Generate Figure 3 showing the scaling behavior of the order parameter with the reduced noise parameter.

Parameters:
- `Ns::Array{Int64, 1}`: Array of particle numbers.
- `ρ::Float64`: Density of the particles.
- `ηcs::Array{Float64, 1}`: Array of critical noise levels.
- `save::Bool`: A boolean indicating whether to save the generated figure. Default is `false`.

Returns:
Nothing. Displays and optionally saves the generated figure.
"""
function fig3(Ns::Array{Int64, 1}, ρ::Float64, ηcs::Array{Float64, 1}, save = false)
    plt3 = plot(xlabel = L"\bar{\eta}", ylabel = L"v_a")

    # Common parameters
    v0 = 0.03; η = 0.1; R = 1; t_steps = 500; Δt = 1; therm_steps = 1000; measure_steps = 500; MCS = 5
     
    for (i,N) in enumerate(Ns)
        ηs = (ηcs[i]-0.1):-0.1:0.1; ηs_bar = (1 .- ηs./ηcs[i])

        L = sqrt(N/ρ)
        params = Parameters(L, N, v0, η, ηs, R, t_steps, Δt, therm_steps, measure_steps, MCS) # Initialize Parameters

        # Initialize State
        x = Array{Float64}(undef, params.N)
        y = Array{Float64}(undef, params.N)
        vx = Array{Float64}(undef, params.N)
        vy = Array{Float64}(undef, params.N)
        θ = Array{Float64}(undef, params.N)

        state = State(x, y, vx, vy, θ) 
        initial_condition(params, state)

        # Initialize Cells
        Nc = floor(Int64, params.L/params.R); rc = NaN
        cells_neighbours = Array{Set{Tuple{Int,Int}}, 2}(undef, Nc, Nc)
        birds_cells = Array{Set{Int}, 2}(undef, Nc, Nc)
        lcell = LinkCellMethod(Nc, rc, cells_neighbours, birds_cells) # LinkCellMethod
        intialize_cells(params, state, lcell)

        # Order Parameter
        φη = Array{Float64}(undef, params.measure_steps)
        φs = Array{Float64}(undef, length(params.ηs))
        δφs = Array{Float64}(undef, length(params.ηs))
        ord_param = OrderParameter(φη, φs, δφs)

        # Check if file exists
        filename = "v_as_N$(N)_fig3.dat"
        # Calculations or loading data if the file exists.
        if isfile(filename)
            println("Loading data for N = $N... ")
            φs, δφs = load_data(filename)
        else
            println("Calculating data for N = $N... ")
            @time φs, δφs = v_vs_η(params, state, lcell, ord_param) #, true)
            save_data(filename, (φs, δφs))
        end

        # Plot
        scatter!(plt3, ηs_bar, φs, xaxis=:log, yaxis=:log, label = L"$N$ = %$N")

        # Linear regression
        linear_regression(log.(ηs_bar), log.(φs), true)
    end
    # Display the plot
    display(plt3)
    
    # Save fig
    save ? savefig(plt3, "fig3.png") : nothing
end

# Critical noise amplitudes to better fit the linear regression
ηcs = [4.8, 4.4, 4.0, 3.4, 3.2] # Chosen by hand
# @time fig3([40, 100, 400, 4000, 10000], 4.0, ηcs)


#----------------------------------------------------------------------
# Other Functions for the Vicsek Model
#----------------------------------------------------------------------
"""
    birds_to_cells2(params::Parameters, state::State, lcell::LinkCellMethod)

Assign particles to cells in the link cell method. But less efficient than birds_to_cells.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.

Returns:
Nothing. Updates the `birds_cells` field of the link cell method object.
"""
function birds_to_cells2(params::Parameters, state::State, lcell::LinkCellMethod)
    for bird in 1:params.N
        i, j = which_cell(state, lcell, bird)
        push!(lcell.birds_cells[i, j], bird)
    end
end

"""
    neighbours_birds_idxs2(params::Parameters, state::State, lcell::LinkCellMethod, bird::Int64)

Find indices of neighboring particles within a certain radius around a given particle using link cell method. But less efficient than neighbours_birds_idxs.

Parameters:
- `params::Parameters`: Parameters of the system.
- `state::State`: State of the system containing particle positions.
- `lcell::LinkCellMethod`: Link cell method object containing cell information.
- `bird::Int64`: Index of the particle.

Returns:
Indices of neighboring particles within the specified radius around the given particle.
"""
function neighbours_birds_idxs2(params::Parameters, state::State, lcell::LinkCellMethod, bird::Int64)
    i, j = which_cell(state, lcell, bird)
    neigh_cells = lcell.cells_neighbours[i, j] # Set of neighbour cells
    neigh_birds = reduce(union, [lcell.birds_cells[cell...] for cell in neigh_cells], init=Set{Int}()) # Set of neighbour birds

    for neigh_b in neigh_birds
        Δx = abs(state.x[bird] - state.x[neigh_b]); Δx_bc = min(Δx, params.L - Δx)
        Δy = abs(state.y[bird] - state.y[neigh_b]); Δy_bc = min(Δy, params.L - Δy)
        r = hypot(Δx_bc, Δy_bc)
        if r > params.R
            delete!(neigh_birds, neigh_b)
        end
    end
    return neigh_birds
end