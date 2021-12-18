class DynamicPlume():
  def __init__(self, 
        sim_dt=0.01,
        birth_rate=1.0,
        env_dt=0.04,
        birthx=1.0, # per-episode puff birth rate sparsity minimum
        birthx_max=1.0, # overall odor puff birth rate sparsity max
        qvar=1.0, # Variance of init. location; higher = more off-plume initializations
        diff_max=0.8, # teacher curriculum
        diff_min=0.4, # teacher curriculum
        verbose=0):
    super(DynamicPlume, self).__init__()

  # Init 
  self.puffs_all = pd.read_csv('constantx5b5_snapshot.csv')
  # self.puffs_all = self.puffs_all[['puff_number', 'time', 'tidx', 'x', 'y', 'radius']]
  self.tidx = self.puffs_all['tidx'].unique().item()
  if birthx_max < 0.99:
    self.puffs_all = self.sparsify(puff_df, birthx_max)
  self.birthx = birthx
  self.steps_per_env_dt = 4 # env_dt/sim_dt hardcoded
  self.birth_rate = birth_rate
  self.wind_y_var = 0.5

  self.diff_min = diff_min
  self.diff_max = diff_max
  self.qvar = qvar

  def sparsify(self, puff_df, birthx=1.0):
    puff_sparsity = np.clip(np.random.uniform(low=birthx, high=1.0), 0.0, 1.0)
    keep_idxs = puff_df['puff_number'].sample(frac=puff_sparsity)
    return puff_df.query("puff_number in @keep_idxs")

  def reset():
    self.wind = [0.5, 0.0]
    self.puffs = self.puffs_all.copy(deep=True)
    # Dynamic birthx for each episode
    self.birthx_ep = np.random.uniform(low=self.birthx, high=1.0)
    self.birthx_ep < 0.99:
      self.puffs = self.sparsify(self.puffs, self.birthx_ep)

  def step():
    # update puffs
    # wind_t = pd.Series({'wind_x': self.wind[0], 'wind_y': self.wind[1]})
    for i in range(self.steps_per_env_dt):
        self.tidx += 1
        self.puffs = sim_utils.manual_integrator(
            self.puffs[['puff_number', 'time', 'tidx', 'x', 'y', 'radius']], 
            pd.Series({'wind_x': self.wind[0], 'wind_y': self.wind[1], 'time':tidx/100}), 
            self.tidx, 
            birth_rate=self.birth_rate*self.birthx_ep, 
            wind_y_var=self.wind_y_var)
    # update wind - none for now


  def get_abunchofpuffs(self, max_samples=300):  
    Z = self.puffs.query(f"tidx == {self.tidx}").loc[:,['x','y']]
    Z = Z.sample(n=max_samples, replace=False) if Z.shape[0] > max_samples else Z
    return Z

  def get_stray_distance(self, agent_location):
    Z = self.get_abunchofpuffs()
    Y = cdist(Z.to_numpy(), np.expand_dims(agent_location,axis=0), metric='euclidean')
    try:
        minY = min(Y) 
    except Exception as ex:
        print(f"Exception: {ex}, t:{self.t_val:.2f}, tidx:{self.tidx}, {Z}")  
        minY = np.array([0])      
    return minY[0] # return float not float-array


  def get_current_wind_xy(self):
    return self.wind

  def get_initial_location(self, loc_algo):
    # assume quantile loc_algo
    q_curriculum = np.random.uniform(self.diff_min, self.diff_max)

    Z = self.get_abunchofpuffs()
    X_pcts = Z['x'].quantile([q_curriculum-0.1, q_curriculum]).to_numpy()
    X_mean, X_var = X_pcts[1], X_pcts[1] - X_pcts[0]
    # print("initial X mean, var, q: ", X_mean, X_var, q_curriculum)
    Y_pcts = Z.query("(x >= (@X_mean - @X_var)) and (x <= (@X_mean + @X_var))")['y'].quantile([0.05,0.5]).to_numpy()
    Y_pcts
    Y_mean, Y_var = Y_pcts[1], min(1, Y_pcts[1] - Y_pcts[0]) # TODO: What was min for?
    # print(Y_mean, Y_var)
    varx = self.qvar 
    loc_xy = np.array([X_mean + varx*X_var*np.random.randn(), 
        Y_mean + varx*Y_var*np.random.randn()]) 
    return loc_xy

