def compute_log_returns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given an input standardized stock price dataframe, transforms
    the dataframe in its relative-valued counterpart:
    - prices are transformed into their log-return
    - volume is transformed into a percent change
    """
    # Copies the input dataframe and updates the column names
    new_df = df.copy()
    # Computes the log-returns for each price column
    price_columns = ["open", "high", "low", "close", "adj_close"]
    for column in price_columns:
        log_price            = np.log(new_df[column])
        log_price_shifted    = np.log(new_df[column].shift(1))
        new_df[column+"_lr"] = log_price - log_price_shifted
    # Computes the volume percent change
    new_df["volume_pct_change"]=new_df["volume"].pct_change()
    return new_df

def compute_bottleneck_distance_persistence_diagrams(
    data:     pd.DataFrame,
    diagrams: list
) -> pd.DataFrame:
    """
    Computes the norm of difference between landscapes.
    """
    # Retrieves the dates corresponding to each diagram +1
    index = data.index[len(data)-len(diagrams)+1:]
    # Computes the exact bottleneck distance
    bottleneck_exact  = lambda ds, i: gd.bottleneck_distance(ds[i-1], ds[i], 0)
    bottleneck_approx = lambda ds, i: gd.bottleneck_distance(ds[i-1], ds[i], 0.001)
    bexact_distances  = [bottleneck_exact(diagrams, i) 
                         for i in range(1, len(diagrams))]
    bapprox_distances = [bottleneck_approx(diagrams, i) 
                         for i in range(1, len(diagrams))]
    # Computes the output
    df_exact  = pd.DataFrame(bexact_distances, 
                             index=index, 
                             columns=["exact_bottleneck_distances"])  
    df_approx = pd.DataFrame(bapprox_distances, 
                             index=index, 
                             columns=["approx_bottleneck_distances"])
    # Plots
    ax = df_approx.plot(figsize=(18,7), 
                   lw      = 0.8, 
                   color   = "orange",
                   alpha   = 0.5,
                   ylabel  = "Distance",
                   title="Approximative bottleneck distances " + \
                   "between consecutive diagrams of k=1, e=0.001")
    ax.axvline(x         = np.where([df_exact.index=="2000-01-10"])[1][0], 
               color     = 'r', 
               linestyle = (0, (3, 5, 1, 5, 1, 5)), 
               label     = 'America Online/Time Warner merger')
    ax.axvline(x         = np.where([df_exact.index=="2008-09-15"])[1][0], 
               color     = 'r', 
               linestyle = '--', 
               label     = 'Lehman Brothers bankruptcy')
    ax.legend()
    ax = df_exact.plot(figsize = (18, 7), 
                 lw      = 0.8, 
                 color   = "blue",
                 alpha   = 0.5,
                 ylabel  = "Distance",
                 title   = "Exact bottleneck distances between consecutive diagrams of k=1")
    ax.axvline(x         = np.where([df_exact.index=="2000-01-10"])[1][0], 
               color     = 'r', 
               linestyle = (0, (3, 5, 1, 5, 1, 5)), 
               label     = 'America Online/Time Warner merger')
    ax.axvline(x         = np.where([df_exact.index=="2008-09-15"])[1][0], 
               color     = 'r', 
               linestyle = '--', 
               label     = 'Lehman Brothers bankruptcy')
    ax.legend()
    return df_exact, df_approx

def compute_norm_difference_persistence_landscapes(
    data:       pd.DataFrame,
    landscapes: list
) -> pd.DataFrame:
    """
    Computes the norm of difference between landscapes.
    """
    # Retrieves the dates corresponding to each landscape +1
    index = data.index[len(data)-len(landscapes)+1:]
    # Computes the norms of the differences
    norm_diffs = lambda ls, i: np.linalg.norm(ls[i]-ls[i-1])
    norm_of_differences = [norm_diffs(landscapes, i) 
                           for i in range(1, len(landscapes))]
    # Computes the output
    df = pd.DataFrame(norm_of_differences, index=index, columns=["norm"])
    ax = df.plot(figsize = (18, 7), 
                 lw      = 0.8, 
                 color   = "green",
                 alpha   = 0.5,
                 ylabel  = "Norm value",
                 title   = "Norm of the difference between consecutive landscapes")
    ax.axvline(x         = np.where([df.index=="2000-01-10"])[1][0], 
               color     = 'r', 
               linestyle = (0, (3, 5, 1, 5, 1, 5)), 
               label     = 'America Online/Time Warner merger')
    ax.axvline(x         = np.where([df.index=="2008-09-15"])[1][0], 
               color     = 'r', 
               linestyle = '--', 
               label     = 'Lehman Brothers bankruptcy')
    ax.legend()
    return df

def compute_persistence_diagram(
    point_cloud:   np.ndarray,
    rips_complex:  bool        = True, 
    print_graph:   bool        = False,
    memory_saving: tuple       = (False, 1)
) -> np.ndarray:
    """
    Given an input point cloud data set, computes the corresponding 
    persistence diagram (only for 1-d loops as in the paper).
    the method relies on using alpha filtration
    """
    # Computes the Vietoris-Rips complex, its barcode and 1-loop diagram
    if rips_complex:
        simplex   = gd.RipsComplex(points = point_cloud)
        simplex   = simplex.create_simplex_tree(max_dimension = 2)
        bar_codes = simplex.persistence()
        if memory_saving[0]:
            simplex = simplex.persistence_intervals_in_dimension(memory_saving[1])
    # Computes the alpha complex, its varcode and 1-loop diagram
    else:
        simplex   = gd.AlphaComplex(points = point_cloud)
        simplex   = simplex.create_simplex_tree()
        bar_codes = [x for x in simplex.persistence() if x[0]<=1]
        if memory_saving[0]:
            simplex = simplex.persistence_intervals_in_dimension(memory_saving[1])
    # prints the persistence diagram graph if requested
    if print_graph: gd.plot_persistence_diagram(bar_codes)
    # the returned diagram comprises the birth and death of 1-loops
    return simplex

def compute_persistence_diagrams(
    data:          pd.DataFrame, 
    w:             int, 
    rips_complex:  bool          = True,
    memory_saving: tuple         = (False, 1)
) -> np.ndarray:
    """
    Given an input time series, computes the corresponding 
    persistence diagram given a shifting window of size w.
    """
    data = data.values
    diagrams = []
    for slc in range(data.shape[0]-w):
        point_cloud = data[slc:slc+w]
        diagram     = compute_persistence_diagram(point_cloud, 
                                                  rips_complex,
                                                  False,
                                                  memory_saving)
        diagrams.append(diagram)
    return diagrams

def compute_persistence_landscape(
    diagram:            np.ndarray,         # diagram range
    endpoints:          list,               # endpoints
    homology_dimension: int         = 1,    # k dimensions
    n_landscapes:       int         = 5,    # m landscapes
    resolution:         int         = 1000, # n nodes
    memory_saving:      bool        = False
) -> np.ndarray:
    """
    Given a persistence diagram of 1D loops of a given
    time series, computes the corresponding persistence landscape.
    Inspired from: https://github.com/MathieuCarriere/sklearn-tda/
                   blob/master/sklearn_tda/vector_methods.py
    """
    # If the diagram is empty, return an empty landscape
    if endpoints[0] == endpoints[1] == 0:
        return np.zeros((n_landscapes, resolution))
    # Renames the min-max range of the given diagram
    diagram_range = endpoints
    # Extracts the homology class from the diagram in case the 
    # computation mode  is not memory-saving. I.e. the dimension
    # class was not pre-fetched at the diagram computation level
    if not memory_saving:
        diagram = diagram.persistence_intervals_in_dimension(homology_dimension)
    # Initializes important variables
    x_range        =  np.linspace(diagram_range[0], 
                                  diagram_range[1],
                                  resolution)
    step           = x_range[1] - x_range[0]
    length_diagram = len(diagram)
    computed_landscapes_at_given_resolution = \
        np.zeros([n_landscapes, resolution])
    computed_y_values = [[] for _ in range(resolution)]
    # Initializes important anonymous functions
    compute_x_subrange = lambda x: int(np.ceil(x/step))
    # Computes the persistence landscape coverage, here
    # the x- and y-axes ranges
    for x, y in diagram:
        # Populates thex-axis range as defined for each 
        # persistence diagram point
        min_point = x - diagram_range[0]
        mid_point = 0.5*(x+y) - diagram_range[0]
        max_point = y - diagram_range[0]
        minimum_x = compute_x_subrange(min_point)
        middle_x  = compute_x_subrange(mid_point)
        maximum_x = compute_x_subrange(max_point)
        # Populates the y-axis values given the computed
        # x-axis range for that part of the resulting landscape
        if minimum_x<resolution and maximum_x>0:
            y_value = diagram_range[0] + minimum_x * step - x
            for z in range(minimum_x, middle_x):
                computed_y_values[z].append(y_value)
                y_value += step
            y_value = y - diagram_range[0] - middle_x * step
            for z in range(middle_x, maximum_x):
                computed_y_values[z].append(y_value)
                y_value -= step
    # Computes for each resolution the corresponding landscape
    for i in range(resolution):
        computed_y_values[i].sort(reverse=True)
        max_range = min(n_landscapes, len(computed_y_values[i]))
        for j in range(max_range):
            computed_landscapes_at_given_resolution[j,i] = \
                computed_y_values[i][j]
    return computed_landscapes_at_given_resolution
    
def compute_persistence_landscapes(
    diagrams:           np.ndarray,         # diagram D
    homology_dimension: int         = 1,    # k dimensions
    n_landscapes:       int         = 5,    # m landscapes
    resolution:         int         = 1000, # n nodes
    memory_saving:      bool        = False
) -> np.ndarray:
    """
    Given a list of persistence diagrams of 1D loops of a given
    time series, computes the corresponding persistence landscapes.
    """
    k    = homology_dimension
    # Declares the anonymous functions helping to compute the
    # diagram endpoints different depending on the memory saving mode
    if memory_saving:
        minp = lambda d: np.min(d) if len(d)>0 else 0
        maxp = lambda d: np.max(d) if len(d)>0 else 0
    else:
        def compute_endpoint(d, minmax):
            d = d.persistence_intervals_in_dimension(k)
            if len(d)>0 and minmax=="min":    return np.min(d)
            elif len(d) >0 and minmax=="max": return np.max(d)
            else:                             return 0
        minp = lambda d: compute_endpoint(d, "min")
        maxp = lambda d: compute_endpoint(d, "max")
    # Transforms all diagrams into landscapes
    landscapes = [
        compute_persistence_landscape(
            diag,                     # diagram D
            [minp(diag), maxp(diag)], # endpoints
            homology_dimension,       # k dimensions
            n_landscapes,             # m landscapes
            resolution,               # n nodes
            memory_saving
        ) for diag in diagrams
    ]
    return landscapes

def compute_persistence_landscape_norms(
    landscapes: list
) -> np.ndarray:
    """
    Given a list/time series of persistence landscape, computes
    the corresponding normalized L1 and L2 time series
    """
    norms_1 = [np.linalg.norm(ls, 1) for ls in landscapes]
    norms_2 = [np.linalg.norm(ls, 2) for ls in landscapes]
    norms_1 = norms_1/np.linalg.norm(norms_1)
    norms_2 = norms_2/np.linalg.norm(norms_2)
    return np.array([norms_1, norms_2]).T

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the column names of the dataframe:
    - lower casing
    - space swapped for underscore
    """
    # Declares useful anonymous function
    format_column_name = lambda x: x.lower().replace(" ", "_")
    # Copies the input dataframe and updates the column names
    new_df = df.copy()
    new_df.columns = list(map(format_column_name, new_df.columns))
    return new_df

def henon_map(
    a:           float, 
    b:           float, 
    n:           int, 
    x0:          float, 
    y0:          float,
    print_graph: bool   = True
) -> np.ndarray:
    """
    Implements a classic hénon map process.
    """
    hm = [(x0, y0)]
    for step in range(0, n):
        y = hm[-1][0]
        x = 1 - a*y**2 + b*hm[-1][1]
        hm.append((x, y))
    hm = np.array(hm)
    if print_graph:
        plt.figure(figsize=(12,2))
        plt.plot(hm[:150,0])
        plt.title("Firsts 150 points of the henon map/attractor system")
        plt.show()
        plt.figure(figsize=(12,12))
        plt.scatter(hm[0,0], hm[0,1], c="red")
        plt.scatter(hm[1:,0], hm[1:,1], s=0.5)
        plt.title("Attractor")
        plt.show()
    return hm[:,0]
    
def noisy_henon_map(
    b:               float, 
    timestep:        float, 
    noise_intensity: float, 
    x0:              float, 
    y0:              float,
    print_graph:     bool   = False
) -> np.ndarray:
    """
    Implements the hénon map with noise modification implemented
    in the paper.
    """
    hm = [(x0, y0)]
    As = [0]
    while As[-1]<1.4:
        random_move = np.random.normal(0,1)*np.sqrt(timestep)
        x = 1 - As[-1]*hm[-1][0]**2 + b*hm[-1][1] + \
            noise_intensity*random_move
        y = hm[-1][0] + noise_intensity*random_move
        next_step = As[-1]+timestep if As[-1]+timestep <=1.4 else 1.4
        As.append(next_step)
        hm.append((x, y))
    hm = np.array(hm)
    if print_graph:
        plt.figure(figsize=(12,2))
        plt.plot(As, hm[:,0])
        plt.title("Henon map/attractor system with Gaussian noise")
        plt.show()
        plt.figure(figsize=(12,12))
        plt.scatter(hm[0,0], hm[0,1], c="red")
        plt.scatter(hm[1:,0], hm[1:,1], s=0.5)
        plt.title("Attractor with Gaussian nosie")
        plt.show()
    return hm[:,0]

def plot_gif_landscapes(
    landscapes:   list, 
    n_landscapes: int,
    resolution:   int
) -> None:
    """
    Given a computed set of landscapes, computes the
    corresponding animated plot of landscapes per window
    """
    Figure = plt.figure()
    # creating a plot
    lines_plotted = [plt.plot([]) for _ in range(n_landscapes)]
    lines_plotted = [l[0] for l in lines_plotted]
    x             = list(range(resolution))
    frames        = len(landscapes)
    plt.xlim(0, resolution)
    plt.ylim(0, 0.002) 
    plt.title("Landscape")
    # function takes frame as an input
    def AnimationFunction(frame):
        plt.title(f"Landscape at window = {frame}")
        for l in range(n_landscapes):
            lines_plotted[l].set_data((x, landscapes[frame][l]))
    anim_created = FuncAnimation(
        Figure, 
        AnimationFunction, 
        frames=frames, 
        interval=10, 
        repeat=False
    )
    video = anim_created.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    # good practice to close the plt object.
    plt.close()

def plot_price_data(
    df_list:       list,  
    legend:        list, 
    target_column: str,
    title:         str
) -> None:
    """
    Given a list of standardized stock price data, plots the
    Adjusted Close value across the whole available timeline.
    """
    plt.figure()
    # Fixes x-ticks interval to c. a year
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
    for df in df_list:
        plt.plot(df["date"].tolist(), 
                 df[[target_column]],
                 linewidth=.5)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend(legend)
    plt.show()

def reproduce_paper_process(
    data:                 pd.DataFrame,
    w_window_size:        int, 
    k_homology_dimension: int,
    m_landscape:          int,
    n_nodes:              int,
    memory_saving:        tuple = (False, 1)
) -> tuple:
    """
    For a given set of financial data, computes the respective persistence
    diagrams and landscapes and display the full L1 and L2 norm persistence
    landscape time series along with a more restricted visualization centered
    around the Dotcom bubble.
    """
    # Abbreviates parameters
    w   = w_window_size
    k   = k_homology_dimension
    m   = m_landscape
    n   = n_nodes
    mem = memory_saving
    # Computes landscapes
    diagrams   = compute_persistence_diagrams(data, w, memory_saving=mem)
    landscapes = compute_persistence_landscapes(diagrams, k, m, n, mem[0])
    # Computes norms
    norms_df   = compute_persistence_landscape_norms(landscapes)
    # Computes and print graph of dot-com bubble
    df = pd.DataFrame(norms_df[2027:3200], 
                      columns = ["L1", "L2"], 
                      index   = data.index[2027+w:3200+w])
    ax = df.plot(figsize = (20, 8), 
                lw       = 0.8, 
                title    = "Persistence landscapes' L1 and L2 " + \
                           "norms towards Dotcom bubble")
    ax.axvline(x         = np.where([df.index=="2000-01-10"])[1][0], 
               color     = 'r', 
               linestyle = (0, (3, 5, 1, 5, 1, 5)), 
               label     = 'America Online/Time Warner merger')
    ax.legend()
    plt.show()
    # Computes the full L1 and L2 time series plot
    df = pd.DataFrame(norms_df, 
                      columns  = ["L1", "L2"],
                      index    = data.index[w:])
    ax = df.plot(figsize = (18, 7), 
                 lw      = 0.8, 
                 ylabel  = "L^p norm value",
                 title   = "Full Persistence landscapes' L1 and L2 norms")
    ax.axvline(x         = np.where([df.index=="2000-01-10"])[1][0], 
               color     = 'r', 
               linestyle = (0, (3, 5, 1, 5, 1, 5)), 
               label     = 'America Online/Time Warner merger')
    ax.axvline(x         = np.where([df.index=="2008-09-15"])[1][0], 
               color     = 'r', 
               linestyle = '--', 
               label     = 'Lehman Brothers bankruptcy')
    ax.legend()
    return diagrams, landscapes, norms_df
    
def reverse_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Revert the order of a dataframe.
    """
    # Copies the input dataframe and updates the column names
    new_df       = df.copy()[::-1]
    new_df.index = range(0, len(df))
    return new_df

def save_data(
    diagrams:   list, 
    landscapes: list, 
    norms:      np.ndarray, 
    w:          int, 
    k:          int, 
    m:          int, 
    n:          int, 
    path:       str
) -> None:
    """
    Saves the produced diagrams, landscapes, and norms output
    from reproduced paper processes.
    """
    p = f"w{w}k{k}m{m}n{n}"
    with open(f"{path}/logretStocksUS_diagrams_{p}", "wb") as f:
        pickle.dump(diagrams, f)
    with open(f"{path}/logretStocksUS_landscapes_{p}", "wb") as f:
        pickle.dump(landscapes, f)
    with open(f"{path}/logretStocksUS_L1L2norms_{p}", "wb") as f:
        pickle.dump(norms, f)

def white_noise_with_gamma_inverse_var(
    n: int = 100
):
    """
    Implements the white noise with gamma-distributed inverse
    variance generation implemented in the paper.
    """
    # Declares the gamma distribution parameters
    alpha      = 8
    beta       = 1
    simulation = None
    # Performs 100 realization of a 4D 100-point cloud dataset
    for realization in range(100):
        # Updates alpha parameter if need be
        if realization > 75:
            alpha -= 0.25
        gamma_variance = np.random.gamma(shape = alpha, scale = beta)
        cloud_set = np.random.normal(
            0, 
            1/np.sqrt(np.random.gamma(shape = alpha, scale = beta)),
            (4, n)
        )
        if simulation is None:
            simulation = cloud_set
        else:
            simulation = np.concatenate([simulation, cloud_set], axis=1)
    return simulation

def white_noise_with_growing_var(
    n_realization: int
) -> dict:
    """
    Implements the white noise with growing variance generation 
    implemented in the paper.
    """
    simulations = {}
    # base variance is grown from 1 to 10, i.e. 10 simulations
    for var in range(1, 11):
        full_simulation = None
        # n realization of 4 time series are performed per
        # simulations
        for n in range(n_realization):
            ts_simulation = None
            for time_series in range(4):
                update_var = var + np.random.uniform(-0.1, 0.1)
                realization = np.random.normal(0, update_var, (1, 100))
                if ts_simulation is None:
                    ts_simulation = realization
                else:
                    ts_simulation = np.concatenate([ts_simulation,
                                                    realization])
            if full_simulation is None:
                full_simulation = ts_simulation
            else:
                full_simulation = np.concatenate([full_simulation,
                                                  ts_simulation], axis=1)
        simulations[var] = full_simulation.reshape(4, -1)
    return simulations
