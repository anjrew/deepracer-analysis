
from io import BytesIO, TextIOWrapper


def get_file(key: str):
    """Downloads a given file as byte array.
    Args:
        key:
            Path to a file on the filesystem.
    Returns: TextIOWrapper
        A bytes object containing the file.
    """

    bytes_io: BytesIO = None
    with open(key, 'rb') as fh:
        bytes_io = BytesIO(fh.read())
        
        
    return TextIOWrapper(
                BytesIO(bytes_io.getvalue()), encoding='utf-8').get()
    

class SimulationLogsIO:
    """ Utilities for loading the logs
    """

    @staticmethod
    def load_single_file(fname, data=None):
        """Loads a single log file and remembers only the SIM_TRACE_LOG lines
        Arguments:
        fname - path to the file
        data - list to populate with SIM_TRACE_LOG lines. Default: None
        Returns:
        List of loaded log lines. If data is not None, it is the reference returned
        and the list referenced has new log lines appended
        """
        if data is None:
            data = []

        with open(fname, 'r') as f:
            for line in f.readlines():
                if "SIM_TRACE_LOG" in line:
                    parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                    data.append(",".join(parts))

        return data
    
    @staticmethod
    def load_buffer(buffer, data=None):
        """Loads a buffered reader and remembers only the SIM_TRACE_LOG lines
        Arguments:
        buffer - buffered reader
        data - list to populate with SIM_TRACE_LOG lines. Default: None
        Returns:
        List of loaded log lines. If data is not None, it is the reference returned
        and the list referenced has new log lines appended
        """
        if data is None:
            data = []

        for line in buffer.readlines():
            if "SIM_TRACE_LOG" in line:
                parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                data.append(",".join(parts))

        return data
    
    @staticmethod
    def load_data(fname):
        """Load all log files for a given simulation
        Looks for all files for a given simulation and loads them. Takes the local training
        into account where in some cases the logs are split when they reach a certain size,
        and given a suffix .1, .2 etc.
        Arguments:
        fname - path to the file
        Returns:
        List of loaded log lines
        """
        from os.path import isfile
        data = []

        i = 1

        while isfile('%s.%s' % (fname, i)):
            SimulationLogsIO.load_single_file('%s.%s' % (fname, i), data)
            i += 1

        SimulationLogsIO.load_single_file(fname, data)

        if i > 1:
            print("Loaded %s log files (logs rolled over)" % i)

        return data
    
    @staticmethod
    def convert_to_pandas(data, episodes_per_iteration=20, stream=None):
        """Load the log data to pandas dataframe
        Reads the loaded log files and parses them according to this format of print:
        stdout_ = 'SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
                self.episodes, self.steps, model_location[0], model_location[1], model_heading,
                self.steering_angle,
                self.speed,
                self.action_taken,
                self.reward,
                self.done,
                all_wheels_on_track,
                current_progress,
                closest_waypoint_index,
                self.track_length,
                time.time())
            print(stdout_)
        Currently only supports 2019 logs but is forwards compatible.
        Arguments:
        data - list of log lines to parse
        episodes_per_iteration - value of the hyperparameter for a given training
        Returns:
        A pandas dataframe with loaded data
        """

        df_list = list()

        # ignore the first two dummy values that coach throws at the start.
        for d in data[2:]:
            parts = d.rstrip().split(",")
            # TODO: this is a workaround and should be removed when logs are fixed
            parts_workaround = 0
            if len(parts) > 17:
                parts_workaround = 1
            episode = int(parts[0])
            steps = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            yaw = float(parts[4])
            steering_angle = float(parts[5])
            speed = float(parts[6])
            try:
                action = int(parts[7])
            except ValueError as e:
                action = -1
            reward = float(parts[8+parts_workaround])
            done = 0 if 'False' in parts[9+parts_workaround] else 1
            all_wheels_on_track = parts[10+parts_workaround]
            progress = float(parts[11+parts_workaround])
            closest_waypoint = int(parts[12+parts_workaround])
            track_len = float(parts[13+parts_workaround])
            tstamp = Decimal(parts[14+parts_workaround])
            episode_status = parts[15+parts_workaround]
            if len(parts) > 16+parts_workaround:
                pause_duration = float(parts[16+parts_workaround])
            else:
                pause_duration = 0.0

            iteration = int(episode / episodes_per_iteration) + 1
            df_list.append((iteration, episode, steps, x, y, yaw, steering_angle, speed,
                            action, reward, done, all_wheels_on_track, progress,
                            closest_waypoint, track_len, tstamp, episode_status, pause_duration))

        header = ['iteration', 'episode', 'steps', 'x', 'y', 'yaw', 'steering_angle',
                  'speed', 'action', 'reward', 'done', 'on_track', 'progress',
                  'closest_waypoint', 'track_len', 'tstamp', 'episode_status', 'pause_duration']

        df = pd.DataFrame(df_list, columns=header)

        if stream is not None:
            df["stream"] = stream

        return df

    @staticmethod
    def load_a_list_of_logs(logs):
        """Loads multiple logs from the list of tuples
        For each file being loaded additional info about the log stream is attached.
        This way one can load multiple simulations for a given period and compare the outcomes.
        This is particularly helpful when comparing multiple evaluations.
        Arguments:
        logs - a list of tuples describing the logs, compatible with the output of
            CloudWatchLogs.download_all_logs
        Returns:
        A pandas dataframe containing all loaded logs data
        """
        full_dataframe = None
        for log in logs:
            eval_data = SimulationLogsIO.load_data(log[0])
            dataframe = SimulationLogsIO.convert_to_pandas(eval_data)
            dataframe['stream'] = log[1]
            if full_dataframe is not None:
                full_dataframe = full_dataframe.append(dataframe)
            else:
                full_dataframe = dataframe

        return full_dataframe.sort_values(
            ['stream', 'episode', 'steps']).reset_index()

    @staticmethod
    def load_pandas(fname, episodes_per_iteration=20):
        """Load from a file directly to pandas dataframe
        Arguments:
        fname - path to the file
        episodes_per_iteration - value of the hyperparameter for a given training
        Returns:
        A pandas dataframe with loaded data
        """
        return SimulationLogsIO.convert_to_pandas(
            SimulationLogsIO.load_data(fname),
            episodes_per_iteration
        )

    @staticmethod
    def normalize_rewards(df):
        """Normalize the rewards to a 0-1 scale
        Arguments:
        df - pandas dataframe with the log data
        """
        from sklearn.preprocessing import MinMaxScaler

        min_max_scaler = MinMaxScaler()
        scaled_vals = min_max_scaler.fit_transform(
            df['reward'].values.reshape(df['reward'].values.shape[0], 1))
        df['reward'] = pd.DataFrame(scaled_vals.squeeze())