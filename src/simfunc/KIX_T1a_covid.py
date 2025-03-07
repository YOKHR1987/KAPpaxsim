# KIX_T1a_covid.py
import copy
import datetime
import heapq
import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simpy
from tqdm import tqdm


def KIX_T1a_covid(
    path: str,
    df_Pax: pd.DataFrame,
    Pt_Z: float,
    N_Z: int,
    Pt_A: float,
    N_A: int,
    Pt_B: float,
    N_B: int,
    Pt_Y1: float,
    N_Y1: int,
    Pt_C1: float,
    N_C1: int,
    Pt_check1: float,
    N_check1: int,
    Pt_rental: float,
    N_rental: int,
    Pt_check2: float,
    N_check2: int,
    Pt_C2: float,
    N_C2: int,
    Pt_C3: float,
    N_C3: int,
    Pt_test: float,
    N_test_slots: int,
    ratio_pax_rental: float,
    ratio_pax_check2: float,
    start_special_pax_ratio: float = 0,
    end_special_pax_ratio: float = 1,
    freq: str = "10min",
    win: int = 1,
    show_loading: bool = True,
    show_graph: bool = False,
    save_graph: bool = False,
    save_xls: bool = False,
    call_n_iter: int = None,
    totalpbar=None,
):
    """Simulate a day of KIX T1 arrival with covid process

    Args:
        path (str): path where results will be stored if save_(xls or grpah) is True
        df_Pax (pd.DataFrame): DataFrame of Pax generated by utils.profiles.generate_arrival_pax
        Pt_Z (float): processing time in SECONDS
        N_Z (int): number of units
        Pt_A (float): [description]
        N_A (int): [description]
        Pt_B (float): [description]
        N_B (int): [description]
        Pt_Y1 (float): [description]
        N_Y1 (int): [description]
        Pt_C1 (float): [description]
        N_C1 (int): [description]
        Pt_check1 (float): [description]
        N_check1 (int): [description]
        Pt_rental (float): [description]
        N_rental (int): [description]
        Pt_check2 (float): [description]
        N_check2 (int): [description]
        Pt_C2 (float): [description]
        N_C2 (int): [description]
        Pt_C3 (float): [description]
        N_C3 (int): [description]
        Pt_test (float): test duration in MINUTES
        N_test_slots (int): number of test slots available
        ratio_pax_rental (float): [description]
        ratio_pax_check2 (float): [description]
        start_special_pax_ratio (float, optional): [description]. Defaults to 0.
        end_special_pax_ratio (float, optional): [description]. Defaults to 1.
        freq (str, optional): frequency of sampling for graphs. Defaults to "10min".
        win (int, optional): window of rolling average for graphs. Defaults to 1.
        show_loading (bool, optional): [description]. Defaults to True.
        show_graph (bool, optional): [description]. Defaults to False.
        save_graph (bool, optional): [description]. Defaults to False.
        save_xls (bool, optional): [description]. Defaults to False.
        call_n_iter (int, optional): [description]. Defaults to None.
        totalpbar ([type], optional): [description]. Defaults to None.

    Returns:
        (
        df_result: full DataFrame of results for each pax
        list_KPI_run: outdated list of KPIs (to be removed eventually)
        dct_hist_wait_time: dictionnary of 'system': [list of wait time for each pax]
        dct_hist_queue_length: dictionnary of 'system': [list of queue length for each pax]
    )
    """
    random.seed(12)  # for reproductibility and smooth optimization

    # change units of Pt
    Pt_Z = Pt_Z / 60
    Pt_A = Pt_A / 60
    Pt_B = Pt_B / 60
    Pt_Y1 = Pt_Y1 / 60
    Pt_C1 = Pt_C1 / 60
    Pt_check1 = Pt_check1 / 60
    Pt_rental = Pt_rental / 60
    Pt_check2 = Pt_check2 / 60
    Pt_C2 = Pt_C2 / 60
    Pt_C3 = Pt_C3 / 60

    # Creating some useful data
    ratio_pax_check1 = 1 - ratio_pax_rental - ratio_pax_check2

    df_Pax["Flight Number"] = df_Pax["Flight Number"].replace(
        ["JX821"], "JX 821"
    )
    df_Pax["Flight Number"] = df_Pax["Flight Number"].replace(
        ["NS*****"], "NS *****"
    )

    df_Pax["airline"] = df_Pax["Flight Number"].apply(lambda x: x.split()[0])
    list_flight = df_Pax["Flight Number"].unique()
    list_airlines = [string for string in df_Pax["airline"].unique()]

    df_Pax["minutes"] = (
        df_Pax["time"].dt.hour.astype(int) * 60
        + df_Pax["time"].dt.minute.astype(int)
        + df_Pax["time"].dt.second.astype(int) / 60
    )
    df_Pax = df_Pax.sort_values(["minutes"]).reset_index(drop=True)

    FREQ = freq
    WINDOW = win

    """
    KIX T1 Int'l arrival
    """

    class arrival_creator(object):
        """
        description
        """

        def __init__(
            self,
            env,
            list_airlines,
            Pt_Z,
            N_Z,
            Pt_A,
            N_A,
            Pt_B,
            N_B,
            Pt_Y1,
            N_Y1,
            Pt_C1,
            N_C1,
            Pt_check1,
            N_check1,
            Pt_rental,
            N_rental,
            Pt_check2,
            N_check2,
            Pt_C2,
            N_C2,
            Pt_C3,
            N_C3,
            Pt_test,
            N_test_slots,
        ):

            self.env = env

            self.Z = simpy.PriorityResource(env, N_Z)
            self.Pt_Z = Pt_Z

            self.A = simpy.PriorityResource(env, N_A)
            self.Pt_A = Pt_A

            self.B = simpy.PriorityResource(env, N_B)
            self.Pt_B = Pt_B

            self.Y1 = simpy.PriorityResource(env, N_Y1)
            self.Pt_Y1 = Pt_Y1

            self.C1 = simpy.PriorityResource(env, N_C1)
            self.Pt_C1 = Pt_C1

            self.check1 = simpy.PriorityResource(env, N_check1)
            self.Pt_check1 = Pt_check1

            self.rental = simpy.PriorityResource(env, N_rental)
            self.Pt_rental = Pt_rental

            self.check2 = simpy.PriorityResource(env, N_check2)
            self.Pt_check2 = Pt_check2

            self.C2 = simpy.PriorityResource(env, N_C2)
            self.Pt_C2 = Pt_C2

            self.C3 = simpy.PriorityResource(env, N_C3)
            self.Pt_C3 = Pt_C3

            self.test_slots = simpy.PriorityResource(env, N_test_slots)
            self.Pt_test = Pt_test

            # dummy resource for the wait_test_result (just to be able to loop after)
            self.wait_test_result = simpy.PriorityResource(
                env, 9999999999999999
            )

        def process_Z(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_Z)

        def process_A(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_A)

        def process_B(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_B)

        def process_Y1(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_Y1)

        def process_C1(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_C1)

        def process_check1(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_check1)

        def process_rental(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_rental)

        def process_check2(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_check2)

        def process_C2(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_C2)

        def process_C3(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_C3)

        def process_test(self, Pax):
            """  """
            yield self.env.timeout(self.Pt_test)

        def process_wait_test_result(self, Pax):
            """ Pax wait in 1 minute increments until the test result becomes available """
            # get Pax index
            index_Pax = int(Pax.split("_")[1])

            # check if result is available from df_result
            result_available = np.isnan(
                df_result.loc[index_Pax, "end_pcr_test_process"]
            )
            while result_available == True:
                yield env.timeout(1)
                result_available = np.isnan(
                    df_result.loc[index_Pax, "end_pcr_test_process"]
                )

    def pcr_test(env, Pax, arr):
        """ create a pcr test and do the test """

        index_Pax = int(Pax.split("_")[1])

        with arr.test_slots.request() as request:
            df_result.loc[index_Pax, "pcr_test_queue_length"] = len(
                arr.test_slots.queue
            )
            df_result.loc[index_Pax, "start_pcr_test_queue"] = env.now
            yield request
            df_result.loc[index_Pax, "end_pcr_test_queue"] = env.now
            yield env.process(arr.process_test(Pax))
            df_result.loc[index_Pax, "end_pcr_test_process"] = env.now

    # ======================================= Passenger journey for each type of Pax=======================================

    def Pax_check1(env, name, arr):
        """
        this pax will do check1 & check2 (no rental, see slides)
        We will do Z,A,B, then start the test process and do Y1,C1,check1,check2,C2,C3
        finally, we will wait for test results
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        # Create a dict of resources
        dct_resources = {
            "Z": arr.Z,
            "A": arr.A,
            "B": arr.B,
            "Y1": arr.Y1,
            "C1": arr.C1,
            "check1": arr.check1,
            "rental": arr.rental,
            "check2": arr.check2,
            "C2": arr.C2,
            "C3": arr.C3,
            "wait_test_result": arr.wait_test_result,
        }
        # Create a dict of processes
        dct_processes = {
            "Z": arr.process_Z,
            "A": arr.process_A,
            "B": arr.process_B,
            "Y1": arr.process_Y1,
            "C1": arr.process_C1,
            "check1": arr.process_check1,
            "rental": arr.process_rental,
            "check2": arr.process_check2,
            "C2": arr.process_C2,
            "C3": arr.process_C3,
            "wait_test_result": arr.process_wait_test_result,
        }

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        # first, Z, A and B
        for process in ["Z", "A", "B"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # then start test process in paralell
        env.process(pcr_test(env, name, arr))

        # then Y1,C1,check1,check2,C2,C3
        for process in ["Y1", "C1", "check1", "check2", "C2", "C3"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # finally wait for test result
        # the only difference is we check the 'count'
        # for number of people waiting instead of queue length
        with arr.wait_test_result.request(priority=2) as request:
            process = "wait_test_result"
            df_result.loc[
                index_Pax, "{}_queue_length".format(process)
            ] = arr.wait_test_result.count
            df_result.loc[
                index_Pax, "start_{}_queue".format(process)
            ] = env.now
            yield request
            # df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now
            yield env.process(dct_processes[process](name))
            # here, we actually consider the process as a queue
            # this is to have a consisten waiting time that makes sense
            df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now

    def Pax_check2(env, name, arr):
        """
        this pax will do check2 (no check1, no rental, see slides)
        We will do Z,A,B, then start the test process and do Y1,C1,check1,check2,C2,C3
        finally, we will wait for test results
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        # Create a dict of resources
        dct_resources = {
            "Z": arr.Z,
            "A": arr.A,
            "B": arr.B,
            "Y1": arr.Y1,
            "C1": arr.C1,
            "check1": arr.check1,
            "rental": arr.rental,
            "check2": arr.check2,
            "C2": arr.C2,
            "C3": arr.C3,
            "wait_test_result": arr.wait_test_result,
        }
        # Create a dict of processes
        dct_processes = {
            "Z": arr.process_Z,
            "A": arr.process_A,
            "B": arr.process_B,
            "Y1": arr.process_Y1,
            "C1": arr.process_C1,
            "check1": arr.process_check1,
            "rental": arr.process_rental,
            "check2": arr.process_check2,
            "C2": arr.process_C2,
            "C3": arr.process_C3,
            "wait_test_result": arr.process_wait_test_result,
        }

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        # first, Z, A and B
        for process in ["Z", "A", "B"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # then start test process in paralell
        env.process(pcr_test(env, name, arr))

        # then Y1,C1,check1,check2,C2,C3
        for process in ["Y1", "C1", "check2", "C2", "C3"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # finally wait for test result
        # the only difference is we check the 'count'
        # for number of people waiting instead of queue length
        with arr.wait_test_result.request(priority=2) as request:
            process = "wait_test_result"
            df_result.loc[
                index_Pax, "{}_queue_length".format(process)
            ] = arr.wait_test_result.count
            df_result.loc[
                index_Pax, "start_{}_queue".format(process)
            ] = env.now
            yield request
            # df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now
            yield env.process(dct_processes[process](name))
            # here, we actually consider the process as a queue
            # this is to have a consisten waiting time that makes sense
            df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now

    def Pax_rental(env, name, arr):
        """
        this pax will do rental (no check1 see slides)
        We will do Z,A,B, then start the test process and do Y1,C1,check1,check2,C2,C3
        finally, we will wait for test results
        """
        df_result.loc[int(name.split("_")[1]), "Pax_ID"] = name
        df_result.loc[int(name.split("_")[1]), "terminal_show_up"] = env.now

        # Create a dict of resources
        dct_resources = {
            "Z": arr.Z,
            "A": arr.A,
            "B": arr.B,
            "Y1": arr.Y1,
            "C1": arr.C1,
            "check1": arr.check1,
            "rental": arr.rental,
            "check2": arr.check2,
            "C2": arr.C2,
            "C3": arr.C3,
            "wait_test_result": arr.wait_test_result,
        }
        # Create a dict of processes
        dct_processes = {
            "Z": arr.process_Z,
            "A": arr.process_A,
            "B": arr.process_B,
            "Y1": arr.process_Y1,
            "C1": arr.process_C1,
            "check1": arr.process_check1,
            "rental": arr.process_rental,
            "check2": arr.process_check2,
            "C2": arr.process_C2,
            "C3": arr.process_C3,
            "wait_test_result": arr.process_wait_test_result,
        }

        airline_code = name.split("_")[2].split()[0]
        index_airline = list_airlines.index(airline_code)
        index_Pax = int(name.split("_")[1])

        # first, Z, A and B
        for process in ["Z", "A", "B"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # then start test process in paralell
        env.process(pcr_test(env, name, arr))

        # then Y1,C1,check1,check2,C2,C3
        for process in ["Y1", "C1", "rental", "check2", "C2", "C3"]:
            with dct_resources[process].request(priority=2) as request:
                df_result.loc[
                    index_Pax, "{}_queue_length".format(process)
                ] = len(dct_resources[process].queue)
                df_result.loc[
                    index_Pax, "start_{}_queue".format(process)
                ] = env.now
                yield request
                df_result.loc[
                    index_Pax, "end_{}_queue".format(process)
                ] = env.now
                yield env.process(dct_processes[process](name))
                df_result.loc[
                    index_Pax, "end_{}_process".format(process)
                ] = env.now

        # finally wait for test result
        # the only difference is we check the 'count'
        # for number of people waiting instead of queue length
        with arr.wait_test_result.request(priority=2) as request:
            process = "wait_test_result"
            df_result.loc[
                index_Pax, "{}_queue_length".format(process)
            ] = arr.wait_test_result.count
            df_result.loc[
                index_Pax, "start_{}_queue".format(process)
            ] = env.now
            yield request
            # df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now
            yield env.process(dct_processes[process](name))
            # here, we actually consider the process as a queue
            # this is to have a consisten waiting time that makes sense
            df_result.loc[index_Pax, "end_{}_queue".format(process)] = env.now

    # ======================================= Passenger generator by flight =======================================

    def Pax_generator(env, arrival, flight, df_Pax_flight, index_total):
        """
        create all the Pax types with their ratios
        """

        # Create initial Pax of the flight
        # global index_vol
        index_vol = 0
        index_total = index_total + index_vol
        N_pax_flight = len(df_Pax_flight["minutes"])
        yield env.timeout(df_Pax_flight["minutes"][index_vol])
        env.process(
            Pax_check1(
                env, "pax_{}_{}_check1".format(index_total, flight), arrival
            )
        )

        # Create the other Paxes
        for index_vol in range(1, N_pax_flight):
            index_total += 1

            yield env.timeout(
                df_Pax_flight["minutes"][index_vol]
                - df_Pax_flight["minutes"][index_vol - 1]
            )
            # generate different types of Pax
            # first, randomly generate the list of index for each type of Pax
            # TO IMPROVE: DO THIS SECTION IN A LOOP OF PAX_TYPES_LIST
            def list_substract(list_1, list_2):
                for element in list_2:
                    if element in list_1:
                        list_1.remove(element)

            start_special_pax_index = int(
                N_pax_flight * start_special_pax_ratio
            )
            end_special_pax_index = int(N_pax_flight * end_special_pax_ratio)

            flight_index_list = [i for i in range(1, N_pax_flight)]
            flight_index_list_orig = flight_index_list.copy()
            flight_index_list = [
                i
                for i in range(start_special_pax_index, end_special_pax_index)
            ]

            random.shuffle(flight_index_list)
            rental_pax_list = flight_index_list[
                0 : int(N_pax_flight * ratio_pax_rental)
            ]
            list_substract(flight_index_list, rental_pax_list)

            random.shuffle(flight_index_list)
            check2_pax_list = flight_index_list[
                0 : int(N_pax_flight * ratio_pax_check2)
            ]
            list_substract(flight_index_list, check2_pax_list)

            # then, generate Pax accordingly
            if index_vol in rental_pax_list:
                env.process(
                    Pax_rental(
                        env,
                        "pax_{}_{}_rental".format(index_total, flight),
                        arrival,
                    )
                )
            elif index_vol in check2_pax_list:
                env.process(
                    Pax_check2(
                        env,
                        "pax_{}_{}_check2".format(index_total, flight),
                        arrival,
                    )
                )

            else:
                env.process(
                    Pax_check1(
                        env,
                        "pax_{}_{}_check1".format(index_total, flight),
                        arrival,
                    )
                )

    # Create dataframe of results
    dummy_list = [np.nan for i in df_Pax.index]

    list_process_all = [
        "Z",
        "A",
        "B",
        "pcr_test",
        "Y1",
        "C1",
        "check1",
        "rental",
        "check2",
        "C2",
        "C3",
        "wait_test_result",
    ]

    L2 = [
        "{}_queue_length",
        "start_{}_queue",
        "end_{}_queue",
        "end_{}_process",
    ]

    L3 = [a.format(b) for b in list_process_all for a in L2]

    list_checkpoints = [
        "Pax_ID",
        "terminal_show_up",
    ]

    list_checkpoints = list_checkpoints + L3

    dct_result = {checkpoint: dummy_list for checkpoint in list_checkpoints}
    df_result = pd.DataFrame(dct_result)

    # Create an environment and start the setup process
    env = simpy.Environment(initial_time=0)
    arrival = arrival_creator(
        env,
        list_airlines,
        Pt_Z,
        N_Z,
        Pt_A,
        N_A,
        Pt_B,
        N_B,
        Pt_Y1,
        N_Y1,
        Pt_C1,
        N_C1,
        Pt_check1,
        N_check1,
        Pt_rental,
        N_rental,
        Pt_check2,
        N_check2,
        Pt_C2,
        N_C2,
        Pt_C3,
        N_C3,
        Pt_test,
        N_test_slots,
    )

    # Generate the Pax

    index_total = 0

    for flight in list_flight:
        # global df_Pax_flight
        df_Pax_flight = (
            df_Pax[df_Pax["Flight Number"] == flight]
            .sort_values(["minutes"])
            .reset_index(drop=True)
        )
        env.process(
            Pax_generator(env, arrival, flight, df_Pax_flight, index_total)
        )
        index_total += len(df_Pax_flight["minutes"])

    # Execute!
    end_time = 1600

    if show_loading == True:
        if call_n_iter is not None and totalpbar is not None:
            with tqdm(
                total=end_time - 1, desc="Simulation running..."
            ) as runpbar:
                for i in range(1, end_time):
                    env.run(until=i)
                    runpbar.update(1)
                    totalpbar.update(1)
        else:
            with tqdm(
                total=end_time - 1, desc="Simulation running..."
            ) as runpbar:
                for i in range(1, end_time):
                    env.run(until=i)
                    runpbar.update(1)

    else:
        env.run(until=1600)

    # ======================================= Results formatting =======================================

    # Manipulate results dat
    # Change to datetinme

    L5 = [
        "start_{}_queue",
        "end_{}_queue",
        "end_{}_process",
    ]

    L6 = [a.format(b) for b in list_process_all for a in L5]

    list_minutes_columns = [
        "terminal_show_up",
    ]

    list_minutes_columns = list_minutes_columns + L6

    def minutes_to_hms(minutes):
        if np.isnan(minutes):
            hms = np.nan
        else:
            hms = "{0:s} {1:0=2d}:{2:0=2d}:{3:0=2d}".format(
                "2020-10-13",
                int((minutes % 1440) // 60),
                int(minutes % 60),
                int((minutes % 1) * 60),
            )
        return hms

    for column in list_minutes_columns:
        df_result[column] = pd.to_datetime(
            df_result[column].apply(lambda x: minutes_to_hms(x))
        )

    # add "Pax_N"
    df_result["Pax_N"] = 1
    # add 'flight_number'
    df_result["flight_number"] = df_result["Pax_ID"].map(
        lambda x: x.split("_")[2]
    )
    # add "STD" eventually, this may be done inside the simulation as we will use STD
    # to determine who has missed his flight and flag them as such
    df_result = (
        pd.merge(
            df_result,
            df_Pax.drop_duplicates("Flight Number")[
                ["Flight Number", "Scheduled Time"]
            ],
            left_on="flight_number",
            right_on="Flight Number",
            how="left",
        )
        .drop(columns="Flight Number")
        .rename(columns={"Scheduled Time": "STD"})
    )

    # Create waiting times
    for process in list_process_all:
        df_result["wait_time_{}".format(process)] = (
            df_result["end_{}_queue".format(process)]
            - df_result["start_{}_queue".format(process)]
        ).fillna(datetime.timedelta(0))

    # for process with start queue but no end queue, set waiting time at 8hrs
    # actually, the queue did not end during sim time so we set the result as high
    for process in list_process_all:
        mask = (pd.isna(df_result["end_{}_queue".format(process)])) & (
            pd.notna(df_result["start_{}_queue".format(process)])
        )

        df_result.loc[
            mask, "wait_time_{}".format(process)
        ] = datetime.timedelta(hours=8)

    # dct plot for graphs by list comprehension
    # they correspond to in/out/queue length/wait time
    list_plot = [
        "start_{}_queue",
        "end_{}_process",
        "{}_queue_length",
        "wait_time_{}",
    ]

    global dct_plot
    dct_plot = {
        key: [plot.format(key) for plot in list_plot]
        for key in list_process_all
    }

    # correction ratio for resampling with sums
    ratio_sampling = pd.to_timedelta("1H") / pd.to_timedelta(FREQ)

    # in
    plt_in = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)["Pax_N"]
            .resample(FREQ)
            .agg(["sum"])
            .rolling(window=WINDOW, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        for key in [*dct_plot]
    ]

    # out
    plt_out = [
        (
            df_result.set_index(dct_plot[key][1], drop=False)["Pax_N"]
            .resample(FREQ)
            .agg(["sum"])
            .rolling(window=WINDOW, center=True)
            .mean()
            .dropna()
            .apply(lambda x: x * ratio_sampling)
        )
        for key in [*dct_plot]
    ]

    # queue length
    plt_queue_length = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)[
                dct_plot[key][2]
            ]
            .resample(FREQ)
            .agg(["max"])
            .rolling(window=WINDOW, center=True)
            .mean()
        )
        for key in [*dct_plot]
    ]

    # queue duration
    plt_queue_duration = [
        (
            df_result.set_index(dct_plot[key][0], drop=False)[
                dct_plot[key][3]
            ]
            .apply(lambda x: x.total_seconds() / 60)
            .resample(FREQ)
            .agg(["max"])
            .rolling(window=WINDOW, center=True)
            .mean()
        )
        for key in [*dct_plot]
    ]
    # histograms of queue duration and queue length

    plt_hist_wait_time = [
        (
            df_result[df_result[dct_plot[key][0]].notnull()][
                dct_plot[key][3]
            ].apply(lambda x: x.total_seconds() / 60)
        )
        for key in [*dct_plot]
    ]

    dct_hist_wait_time = {
        key: (
            df_result[df_result[dct_plot[key][0]].notnull()][
                dct_plot[key][3]
            ].apply(lambda x: x.total_seconds() / 60)
        )
        for key in [*dct_plot]
    }

    dct_hist_queue_length = {
        key: (
            df_result[df_result[dct_plot[key][0]].notnull()][dct_plot[key][2]]
        )
        for key in [*dct_plot]
    }

    # ======================================= Plotting =======================================
    n_graph = len([*dct_plot])
    if show_graph == True:
        # plot param
        xmin = pd.to_datetime("2020-10-13 00:00:00")
        xmax = pd.to_datetime("2020-10-14 00:00:00")
        plt.rcParams.update({"figure.autolayout": True})
        hours = mdates.HourLocator(interval=1)
        half_hours = mdates.MinuteLocator(byminute=[0, 30], interval=1)
        h_fmt = mdates.DateFormatter("%H:%M:%S")

        # plotting
        widths = [4, 1]
        gs_kw = dict(width_ratios=widths)

        fig = plt.figure(figsize=(16, 4 * n_graph))

        axs = fig.subplots(n_graph, 2, squeeze=True, gridspec_kw=gs_kw)
        ax2 = [axs[i, 0].twinx() for i in range(n_graph)]

        # plot for all processes, except 'wait for counter opening'
        for i in range(n_graph):
            axs[i, 0].plot(plt_in[i], label="in", lw=2)
            axs[i, 0].plot(plt_out[i], label="out", lw=2)
            axs[i, 0].plot(
                plt_queue_length[i], label="queue length", ls="--", lw=1
            )
            ax2[i].plot(
                plt_queue_duration[i],
                label="queue duration",
                color="r",
                ls="--",
                lw=1,
            )

            sns.histplot(plt_hist_wait_time[i], ax=axs[i, 1], bins=30)

            axs[i, 0].set(
                ylabel="Pax/hr or Pax",
                title=[*dct_plot][i],
                xlim=[xmin, xmax],
            )
            axs[i, 0].set_ylim(bottom=0)
            axs[i, 0].xaxis.set_major_locator(hours)
            axs[i, 0].xaxis.set_major_formatter(h_fmt)
            axs[i, 0].xaxis.set_minor_locator(half_hours)
            axs[i, 0].legend(loc="upper left")

            axs[i, 1].set_xlim(left=0)

            ax2[i].legend(loc="upper right")
            ax2[i].set(ylabel="waiting time [min]")
            ax2[i].set_ylim(bottom=0)
            ax2[i].spines["right"].set_color("r")
            ax2[i].tick_params(axis="y", colors="r")
            ax2[i].yaxis.label.set_color("r")

        # remove ticks labels for all grows except last
        for i in range(n_graph - 1):
            ax2[i].tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
            axs[i, 0].tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

        # format last row graphs xticks labels
        labels = axs[n_graph - 1, 0].get_xticklabels()
        plt.setp(labels, rotation=45, horizontalalignment="right")
        axs[n_graph - 1, 0].set(xlabel="time")

        if save_graph == True:
            plt.savefig(path + "/KIX_T1_dep.jpg")

    # set parameters in a DataFrame
    # maybe there is a more elegant way to do that
    dct_param_run = {
        "path": path,
        "Pt_Z": Pt_Z,
        "N_Z": N_Z,
        "Pt_A": Pt_A,
        "N_A": N_A,
        "Pt_B": Pt_B,
        "N_B": N_B,
        "Pt_Y1": Pt_Y1,
        "N_Y1": N_Y1,
        "Pt_C1": Pt_C1,
        "N_C1": N_C1,
        "Pt_check1": Pt_check1,
        "N_check1": N_check1,
        "Pt_rental": Pt_rental,
        "N_rental": N_rental,
        "Pt_check2": Pt_check2,
        "N_check2": N_check2,
        "Pt_C2": Pt_C2,
        "N_C2": N_C2,
        "Pt_C3": Pt_C3,
        "N_C3": N_C2,
        "Pt_test": Pt_test,
        "N_test_slots": N_test_slots,
        "ratio_pax_rental": ratio_pax_rental,
        "ratio_pax_check2": ratio_pax_check2,
        "start_special_pax_ratio": start_special_pax_ratio,
        "end_special_pax_ratio": end_special_pax_ratio,
        "freq": freq,
        "win": win,
        "show_loading": show_loading,
        "show_graph": show_graph,
        "save_graph": save_graph,
        "save_xls": save_xls,
        "call_n_iter": call_n_iter,
    }

    df_param_run = pd.DataFrame(dct_param_run, index=[0])
    if show_graph == True:
        print(df_param_run)

    if save_xls == True:
        # write set results to Excel
        writer = pd.ExcelWriter(
            path + r"\run_results.xlsx",
            engine="xlsxwriter",
        )
        df_result.to_excel(writer, sheet_name="results")
        df_param_run.transpose().to_excel(writer, sheet_name="parameters")
        df_Pax.to_excel(writer, sheet_name="Pax_input")

        writer.save()

    list_kpi_queue_length = [
        list(plt_queue_length[i]["max"].replace(np.nan, 0))
        for i in range(n_graph)
    ]
    list_kpi_wait_time = [list(plt_hist_wait_time[i]) for i in range(n_graph)]

    kpi_queue_length = [
        min(
            heapq.nlargest(
                int(len(list_kpi_queue_length[i]) / 90) + 1,
                list_kpi_queue_length[i],
            )
        )
        for i in range(n_graph)
    ]
    kpi_wait_time = [
        min(
            heapq.nlargest(
                int(len(list_kpi_wait_time[i]) / 90) + 1,
                list_kpi_wait_time[i],
            )
        )
        for i in range(n_graph)
    ]

    list_KPI_run = [
        [kpi_queue_length[i], kpi_wait_time[i]] for i in range(n_graph)
    ]

    return (
        df_result,
        list_KPI_run,
        dct_hist_wait_time,
        dct_hist_queue_length,
    )


def univariate_cost_function_generator_T1a_covid_N(
    variable_string,  # eg. N_kiosk
    target_wait_time,  # target waiting time in minutes for considered system
    dct_param_T1a,  # includes df_Pax and df_Counters
    call_n_iter=None,
    totalpbar=None,
):
    """
    this function generates a univariate cost function for T1 arrival

    the variable must be selected: which parameter should the function optimize
    => for now we will do the N only
    => after we should also make the following possible:
        - traffic (easy)
        - areas
    """

    def cost_function_T1a_covid_N(
        x,  # value of variable for which cost function should be evaluated
        variable_string=variable_string,  # variable that should be optimized, eg. N_Z
        target_wait_time=target_wait_time,  # target waiting time in minutes
        dct_param_T1a=dct_param_T1a,
    ):
        """
        calculate the cost function for KIX T1 arrival
        """
        # regenerate the system name corresponding to dct_hist_wait_time
        if variable_string == "N_test_slots":
            system_string = "wait_test_result"
        else:
            system_string = variable_string.split("_")[1]

        # pass the variable for the parameter to be optimized
        dct_param_T1a[variable_string] = x

        # run the model and get the wait_time and queue_length dicts
        (
            _,
            _,
            dct_hist_wait_time,
            dct_hist_queue_length,
        ) = KIX_T1a_covid(**dct_param_T1a)

        # caculate cost
        cost_wait_time_run = (
            dct_hist_wait_time[system_string].quantile(q=0.90)
            - target_wait_time
        ) ** 2

        # correction if:

        # if top90% pax do not wait, then we need to penalize high number of N
        if dct_hist_wait_time[system_string].quantile(q=0.90) == 0:
            cost_wait_time_run += x / 10000

        # if the top90% Pax waits 8hrs or more, we need to consider the mean waiting time
        if dct_hist_wait_time[system_string].quantile(q=0.90) >= 13.9 * 60:
            cost_wait_time_run += (
                (dct_hist_wait_time[system_string].mean() - target_wait_time)
                ** 2
            ) / 10000

        return cost_wait_time_run

    return cost_function_T1a_covid_N
