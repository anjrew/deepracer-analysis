import get_file

def load_robomaker_logs(type: LogType = LogType.TRAINING, force: bool = False):
    """ Method that loads DeepRacer robomaker log into a dataframe.
    The method will load in all available workers and iterations from one training run.
    Args:
        type:
            By specifying `LogType` as either `TRAINING`, `EVALUATION` or `LEADERBOARD`
            then different logs will be loaded. In the case of `EVALUATION` or `LEADERBOARD`
            multiple logs may be loaded.
        force:
            Enables the reloading of logs. If `False` then loading will be blocked
            if the dataframe is already populates.
    """

    if type == LogType.TRAINING:

        raw_data = get_file(self.fh.training_robomaker_log_path)

        parse_robomaker_metadata(raw_data)

        episodes_per_iteration = hyperparameters["num_episodes_between_training"]

        data: list[str] = SimulationLogsIO.load_buffer(raw_data)
        self.df = SimulationLogsIO.convert_to_pandas(data, episodes_per_iteration)
        self.active = LogType.TRAINING

    else:
        dfs = []

        if type == LogType.EVALUATION:
            submissions = self.fh.list_files(check_exist=True,
                                                filterexp=self.fh.evaluation_robomaker_log_path)
            splitRegex = re.compile(self.fh.evaluation_robomaker_split)

        elif type == LogType.LEADERBOARD:
            submissions = self.fh.list_files(check_exist=True,
                                                filterexp=self.fh.leaderboard_robomaker_log_path)
            splitRegex = re.compile(self.fh.leaderboard_robomaker_log_split)

        for i, log in enumerate(submissions):
            path_split = splitRegex.search(log)
            raw_data = get_file(log)

            if i == 0:
                self._parse_robomaker_metadata(raw_data)

            data = SimulationLogsIO.load_buffer(raw_data)
            dfs.append(SimulationLogsIO.convert_to_pandas(data, stream=path_split.groups()[0]))

        self.df = pd.concat(dfs, ignore_index=True)
        self.active = type

def parse_robomaker_metadata(raw_data: bytes):

    outside_hyperparams = True
    hyperparameters_string = ""
    hyperparameters: dict

    data_wrapper = TextIOWrapper(BytesIO(raw_data), encoding='utf-8')

    for line in data_wrapper.readlines():
        if outside_hyperparams:
            if "Using the following hyper-parameters" in line:
                outside_hyperparams = False
        else:
            hyperparameters_string += line
            if "}" in line:
                hyperparameters = json.loads(hyperparameters_string)
                break

    data_wrapper.seek(0)

    if hyperparameters is None:
        raise Exception("Cound not load hyperparameters. Exiting.")

    for line in data_wrapper.readlines():
        if "ction space from file: " in line:
            self._action_space = json.loads(line.split("file: ")[1].replace("'", '"'))

    data_wrapper.seek(0)

    regex = r'Sensor list (\[[\'a-zA-Z, _-]+\]), network ([a-zA-Z_]+), simapp_version ([\d.]+)'
    agent_and_network = {}
    for line in data_wrapper.readlines():
        if " * /WORLD_NAME: " in line:
            agent_and_network["world"] = line[:-1].split(" ")[-1]
        elif "Sensor list ['" in line:
            m = re.search(regex, line)

            agent_and_network["sensor_list"] = json.loads(m.group(1).replace("'", '"'))
            agent_and_network["network"] = m.group(2)
            agent_and_network["simapp_version"] = m.group(3)

            self._agent_and_network = agent_and_network
            break

    data_wrapper.seek(0)
    
    return dict({ 
                '' 
                 })

