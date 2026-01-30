import math
import networkx as nx
import random
from enum import Enum
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from model.directed_network_space import DirectedNetworkSpace
import datetime
import csv


class State(Enum):
    NEUTRAL = 0
    SHARED = 1
    REFUSED = 2


class GraphType(Enum):
    DEGREE_SEQUENCE = 0
    ERDOS_RENYI = 1
    BARABASI_ALBERT = 2
    COMPLETELY_CONNECTED = 3


class CentralityView(Enum):
    BETWEENNESS = 0
    CLOSENESS = 1
    DEGREE = 2
    HARMONIC = 3


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_shared(model):
    return number_state(model, State.SHARED)


def number_neutral(model):
    return number_state(model, State.NEUTRAL)


def number_refused(model):
    return number_state(model, State.REFUSED)


def number_aged(model, min_age, max_age):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.age in range(min_age, max_age + 1))


def number_13_17(model):
    return number_aged(model, 13, 17)


def number_18_24(model):
    return number_aged(model, 18, 24)


def number_25_34(model):
    return number_aged(model, 25, 34)


def number_35_49(model):
    return number_aged(model, 35, 49)


def number_50(model):
    return number_aged(model, 50, 50)


def number_education_level(model, education_level):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.education == education_level)


def number_no_diploma(model):
    return number_education_level(model, "No diploma")


def number_secondary_diploma(model):
    return number_education_level(model, "Secondary diploma")


def number_post_secondary_diploma(model):
    return number_education_level(model, "Post-secondary diploma")


def centrality_colour(agent):
    match agent.relative_centrality:
        case 100:
            return "#000066"
        case agent.relative_centrality if agent.relative_centrality >= 75:
            return "#0000ff"
        case agent.relative_centrality if 75 > agent.relative_centrality >= 50:
            return "#0099ff"
        case agent.relative_centrality if 50 > agent.relative_centrality >= 25:
            return "#33ccff"
        case _:
            return "#66ffff"


def get_graph_type(graph_type):
    match graph_type:
        case "Degree Sequence":
            return GraphType.DEGREE_SEQUENCE
        case "Erdős–Rényi":
            return GraphType.ERDOS_RENYI
        case "Barabási–Albert":
            return GraphType.BARABASI_ALBERT
        case _:
            return GraphType.COMPLETELY_CONNECTED


def get_centrality_view(centrality_view):
    match centrality_view:
        case 'Betweenness':
            return CentralityView.BETWEENNESS
        case 'Closeness':
            return CentralityView.CLOSENESS
        case 'Degree':
            return CentralityView.DEGREE
        case _:
            return CentralityView.HARMONIC


class SocialMediaNetwork(Model):
    """This is a customisable agent-based social media network OSNModel. The purpose of the OSNModel is to simulate the
     propagation of misinformation on social media. The OSNModel has a variety of parameters which can be customised."""

    def __init__(
            self,
            num_nodes=100,
            share_chance=50,
            refuse_chance=100,
            num_initial_nodes=1,
            directed=False,
            initial_nodes_high_degree=True,
            centrality_view='Degree',
            graph_type='Degree sequence',
            power_law=2.5,
            erdos_renyi_edge_probability=0.05,
            barabasi_albert_num_edges=3,
            perc_13_17=7,
            perc_18_24=17,
            perc_25_34=38,
            perc_35_49=21,
            perc_50=17,
            perc_no_diploma=12,
            perc_secondary_diploma=42,
            perc_post_secondary_diploma=46,
            accuracy_nudge_intervention=0,
    ):

        self.num_nodes = num_nodes
        self.share_chance = share_chance / 100
        self.refuse_chance = refuse_chance / 100
        self.num_initial_nodes = num_initial_nodes
        self.directed = directed
        self.initial_nodes_high_degree = initial_nodes_high_degree
        self.centrality_view = get_centrality_view(centrality_view)
        self.graph_type = get_graph_type(graph_type)
        self.power_law = power_law
        self.erdos_renyi_edge_probability = erdos_renyi_edge_probability
        self.barabasi_albert_num_edges = barabasi_albert_num_edges
        self.perc_13_17 = perc_13_17 / 100
        self.perc_18_24 = perc_18_24 / 100
        self.perc_25_34 = perc_25_34 / 100
        self.perc_35_49 = perc_35_49 / 100
        self.perc_50 = perc_50 / 100
        self.perc_no_diploma = perc_no_diploma / 100
        self.perc_secondary_diploma = perc_secondary_diploma / 100
        self.perc_post_secondary_diploma = perc_post_secondary_diploma / 100
        self.accuracy_nudge_intervention = accuracy_nudge_intervention / 100

        self.output_file = f"output\\OSNModel{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}.csv"
        self.G = self.create_graph()
        self.grid = DirectedNetworkSpace(self.G)
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            {
                "Shared": number_shared,
                "Neutral": number_neutral,
                "Refused": number_refused,
                "Aged 13-17": number_13_17,
                "Aged 18-24": number_18_24,
                "Aged 25-34": number_25_34,
                "Aged 35-49": number_35_49,
                "Aged 50+": number_50,
                "No diploma": number_no_diploma,
                "Secondary diploma": number_secondary_diploma,
                "Post-secondary diploma": number_post_secondary_diploma,
            }
        )

        self.centrality = self.get_centrality()
        self.followers = self.get_followers()

        for index, node in enumerate(self.G.nodes()):  # For each node
            a = UserAgent(  # Create a new User Agent with:
                unique_id=index,  # A unique ID = the number of the node
                model=self,  # Pass a copy of the network model
                initial_state=State.NEUTRAL,  # Set the initial state to neutral
                followers=self.followers[node][1],  # Pass the node's follower IDs
                age=None,  # Set age to None for now
                education=None,  # Set education to None for now
                betweenness_centrality=round(self.centrality[0][index], 2),
                closeness_centrality=round(self.centrality[1][index], 2),
                degree_centrality=round(self.centrality[2][index], 2),
                harmonic_centrality=round(self.centrality[3][index], 2),
                # Set the respective centrality scores of the node
                relative_centrality=self.get_relative_centrality(index),
                # Set the relative centrality of the node
                shared_from=None  # Set shared_from to None for now
            )
            self.schedule.add(a)  # Add the node to the scheduler
            self.grid.place_agent(a, node)  # Place the node on the grid

        self.set_ages()
        self.set_educations()

        initial_nodes = self.get_initial_nodes()
        for a in self.grid.get_cell_list_contents(initial_nodes):
            a.state = State.SHARED

        self.running = True
        self.datacollector.collect(self)
        self.output_network()
        self.output_state()

    def output_network(self):
        with open(self.output_file, 'w', newline='') as file:
            # Open a new file in write mode with the name generated when creating the model
            writer = csv.writer(file)
            # Create a new CSV writer
            writer.writerows(
                [["Number of nodes", "Share chance", "Refuse chance", "Number of initial nodes", "Directed",
                  "High degree initial node(s)", "Centrality view", "Graph type", "Power law (degree)",
                  "Edge probability (Erdos-Renyi)", "Number of edges (Barabasi-Albert)", "No. 13-17",
                  "No. 18-24", "No. 25-34", "No. 35-49", "No. 50+", "No. with no diploma",
                  "No. with secondary diploma", "No. with post-secondary diploma",
                  "Accuracy-nudge intervention chance"],
                 [self.num_nodes, self.share_chance, self.refuse_chance, self.num_initial_nodes,
                  self.directed, self.initial_nodes_high_degree, self.centrality_view,
                  self.graph_type, self.power_law, self.erdos_renyi_edge_probability,
                  self.barabasi_albert_num_edges, number_13_17(self), number_18_24(self), number_25_34(self),
                  number_35_49(self), number_50(self), number_no_diploma(self),
                  number_secondary_diploma(self), number_post_secondary_diploma(self),
                  self.accuracy_nudge_intervention],
                 ["No. neutral", "No. shared", "No. refused"]])
            # Write each parameter name to a cell in the first row
            # Write each parameter value to a cell in the second row
            # Write "No. neutral", "No. shared", "No. refused" to the third row

    def output_state(self):
        with open(self.output_file, 'a', newline='') as file:
            # Open the previously created file in append mode
            writer = csv.writer(file)
            # Create a new CSV writer
            writer.writerow([number_neutral(self), number_shared(self), number_refused(self)])
            # Write the number of neutral, shared, and refused nodes to the next row

    def set_ages(self):
        sum_nodes = num_13_17 = int(self.perc_13_17 * self.num_nodes)

        num_18_24 = int(self.perc_18_24 * self.num_nodes)
        if sum_nodes + num_18_24 > self.num_nodes:
            num_18_24 = self.num_nodes - sum_nodes
        sum_nodes += num_18_24

        num_25_34 = int(self.perc_25_34 * self.num_nodes)
        if sum_nodes + num_25_34 > self.num_nodes:
            num_25_34 = self.num_nodes - sum_nodes
        sum_nodes += num_25_34

        num_35_49 = int(self.perc_35_49 * self.num_nodes)
        if sum_nodes + num_35_49 > self.num_nodes:
            num_35_49 = self.num_nodes - sum_nodes
        sum_nodes += num_35_49

        num_50 = int(self.perc_50 * self.num_nodes)
        if sum_nodes + num_50 > self.num_nodes:
            num_50 = self.num_nodes - sum_nodes
        sum_nodes += num_50

        if sum_nodes < self.num_nodes:
            num_18_24 += self.num_nodes - sum_nodes

        self.set_age(num_13_17, 13, 17)
        self.set_age(num_18_24, 18, 24)
        self.set_age(num_25_34, 25, 34)
        self.set_age(num_35_49, 35, 49)
        self.set_age(num_50, 50, 50)

    def set_age(self, num, min_age, max_age):
        if num > 0:
            for node in random.sample(list((node for node in
                                            self.grid.get_cell_list_contents(list(range(0, self.num_nodes)))
                                            if node.age is None)), num):
                node.age = random.randint(min_age, max_age)

    def set_educations(self):
        sum_nodes = num_no_diploma = int(self.perc_no_diploma * self.num_nodes)

        num_secondary_diploma = int(self.perc_secondary_diploma * self.num_nodes)
        if sum_nodes + num_secondary_diploma > self.num_nodes:
            num_secondary_diploma = self.num_nodes - sum_nodes
        sum_nodes += num_secondary_diploma

        num_post_secondary_diploma = int(self.perc_post_secondary_diploma * self.num_nodes)
        if sum_nodes + num_post_secondary_diploma > self.num_nodes:
            num_post_secondary_diploma = self.num_nodes - sum_nodes
        sum_nodes += num_post_secondary_diploma

        if sum_nodes < self.num_nodes:
            num_secondary_diploma += self.num_nodes - sum_nodes

        self.set_education(num_post_secondary_diploma, "Post-secondary diploma")
        self.set_education(num_secondary_diploma, "Secondary diploma")
        self.set_education(num_no_diploma, "No diploma")

    def set_education(self, num, education_string):
        if education_string != "No diploma":
            num_uneducated_nodes = len(list((node for node in
                                             self.grid.get_cell_list_contents(list(range(0, self.num_nodes)))
                                             if node.education is None)))
            if num_uneducated_nodes - int(self.perc_13_17 * self.num_nodes) > num:
                for node in random.sample(list((node for node in
                                                self.grid.get_cell_list_contents(list(range(0, self.num_nodes)))
                                                if node.education is None and node.age not in range(13, 18))), num):
                    node.education = education_string
            else:
                for node in random.sample(list((node for node in
                                                self.grid.get_cell_list_contents(list(range(0, self.num_nodes)))
                                                if node.education is None)), num):
                    node.education = education_string
        else:
            for node in random.sample(list((node for node in
                                            self.grid.get_cell_list_contents(list(range(0, self.num_nodes)))
                                            if node.education is None)), num):
                node.education = education_string

    def get_relative_centrality(self, index):
        match self.centrality_view:
            case CentralityView.BETWEENNESS:
                centrality_index = 0
            case CentralityView.CLOSENESS:
                centrality_index = 1
            case CentralityView.DEGREE:
                centrality_index = 2
            case _:
                centrality_index = 3
        # Select which centrality type to calculate the relative score for
        # (This depends on the CentralityView parameter selected by the user)
        val = (max(self.centrality[centrality_index].values()) - min(self.centrality[centrality_index].values()))
        if val != 0:
            return round(self.centrality[centrality_index][index] / val * 100, 2)
        else:
            return self.centrality[centrality_index][index]

    def create_graph(self):
        match self.graph_type:
            case GraphType.DEGREE_SEQUENCE:
                return self.create_degree_sequence_graph()
            case GraphType.ERDOS_RENYI:
                return self.create_erdos_renyi_graph()
            case GraphType.BARABASI_ALBERT:
                return self.create_barabasi_albert_graph()
            case _:
                return self.create_completely_connected_graph()

    def create_degree_sequence(self, power_law_value):
        while True:
            degree_sequence = []
            while len(degree_sequence) < self.num_nodes:
                degree_value = int(nx.utils.powerlaw_sequence(1, power_law_value)[0])
                if degree_value != 0:
                    degree_sequence.append(degree_value)
            if sum(degree_sequence) % 2 == 0:
                break
        return degree_sequence

    def create_degree_sequence_graph(self):
        degree_sequence = self.create_degree_sequence(self.power_law)
        if self.directed:
            degree_in = degree_out = []
            for degree in degree_sequence:
                degree_in.append(round(degree / 2))
                degree_out.append(round(degree / 2))
            configuration_model_graph = nx.directed_configuration_model(degree_in, degree_out)
        else:
            configuration_model_graph = nx.configuration_model(degree_sequence)

        configuration_model_graph = nx.Graph(configuration_model_graph)
        configuration_model_graph.remove_edges_from(nx.selfloop_edges(configuration_model_graph))

        if not nx.is_connected(configuration_model_graph):
            sub_graph_list = list(
                (configuration_model_graph.subgraph(c) for c in nx.connected_components(configuration_model_graph)))

            target_sub_graph = sub_graph_list[0]
            for sub_graph in sub_graph_list:
                if len(sub_graph) > len(target_sub_graph):
                    target_sub_graph = sub_graph

            unfrozen_graph = nx.Graph(target_sub_graph)

            index = 0
            mapping = {}
            for node in unfrozen_graph:
                mapping.update({int(node): int(index)})
                index += 1
            relabelled_graph = nx.relabel_nodes(unfrozen_graph, mapping)

            num_net_nodes = len(relabelled_graph)
            for i in range(num_net_nodes, self.num_nodes):
                relabelled_graph.add_edge(int(random.choice(list(relabelled_graph))), i)

            graph = relabelled_graph
        else:
            graph = configuration_model_graph

        return graph

    def create_erdos_renyi_graph(self):
        return nx.erdos_renyi_graph(self.num_nodes, self.erdos_renyi_edge_probability, directed=self.directed)

    def create_barabasi_albert_graph(self):
        if self.directed:
            edges = list(nx.barabasi_albert_graph(self.num_nodes, self.barabasi_albert_num_edges, seed=0).edges())
            directed_edges = [(edges[i][0], edges[i][1]) if i % 2 == 0 else
                              (edges[i][1], edges[i][0]) for i in list(range(len(edges)))]
            graph = nx.DiGraph()
            graph.add_edges_from(directed_edges)
        else:
            graph = nx.barabasi_albert_graph(self.num_nodes, self.barabasi_albert_num_edges)
        return graph

    def create_completely_connected_graph(self):
        return nx.complete_graph(self.num_nodes)

    def get_centrality(self):
        betweenness_cen = nx.betweenness_centrality(self.G)
        # Get a dictionary of each node and its betweenness score
        closeness_cen = nx.closeness_centrality(self.G)
        # Get a dictionary of each node and its closeness score
        degree_cen = nx.degree_centrality(self.G)
        # Get a dictionary of each node and its degree score
        harmonic_cen = nx.harmonic_centrality(self.G)
        # Get a dictionary of each node and its harmonic score
        return [betweenness_cen, closeness_cen, degree_cen, harmonic_cen]
        # Return a list of each dictionary

    def get_followers(self):
        followers = []  # Create an empty list
        if self.directed:  # If the network is directed
            for node in self.G.nodes:
                followers.append((node, set()))
                # For each node, append a dictionary with key = the node's ID and an empty value
            for edge in self.G.edges:
                followers[edge[0]][1].add(edge[1])
                # For each edge, add the target node to the source node's followers
        else:  # If the network is undirected
            for node in self.G.nodes:
                followers.append((node, {n for n in self.G.neighbors(node)}))
                # Use the built-in neighbors function to get all nodes connected to the node
        return followers

    def get_initial_nodes(self):
        shared_list = []  # List of initially shared nodes
        if self.initial_nodes_high_degree:  # If High Degree Initial Shares is on:
            num_followers = sorted(self.followers, key=lambda x: len(x[1]), reverse=True)
            # Get list of nodes sorted by their followers
            for index in range(0, self.num_initial_nodes):
                shared_list.append(num_followers[index][0])
                # Add the top n nodes to the list
        else:  # If High Degree Initial Shares is off:
            shared_list = random.sample(list((node for node in [n[0] for n in self.followers if len(n[1]) > 0])),
                                        self.num_initial_nodes)
            # Add n random nodes where followers > 0 to the list
        return shared_list

    def shared_neutral_ratio(self):
        try:
            return number_state(self, State.SHARED) / number_state(
                self, State.NEUTRAL
            )
        except ZeroDivisionError:
            return math.inf

    def refused_neutral_ratio(self):
        try:
            return number_state(self, State.REFUSED) / number_state(
                self, State.NEUTRAL
            )
        except ZeroDivisionError:
            return math.inf

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.output_state()

    def run_model(self, n):
        for i in range(n):
            self.step()


def get_age_multiplier(following_agent):
    match int(following_agent.age):  # If user is aged:
        case n if n in range(45, 51):  # Between 45-50
            return round(random.uniform(1.14, 2.05), 2)
            # Multiply the chance of sharing by between 1.14 times and 2.05 times
        case n if n in range(30, 45):  # Between 30 and 44
            return 0.77
            # Multiply the chance of sharing by 0.77 times
        case _:  # Otherwise:
            return 0.38
            # Multiply the chance of sharing by 0.38 times


def get_education_multiplier(following_agent):
    match following_agent.education:  # If user has education level:
        case "No diploma":  # No diploma
            return 1.165  # Multiply sharing chance by 1.165 times
        case "Secondary diploma":
            return 1  # Does not affect sharing chance
        case _:
            return 0.835  # Multiple sharing chance by 0.835 times


def intervention(following_agent):
    if random.random() < following_agent.model.accuracy_nudge_intervention:
        # Check if the accuracy-nudge intervention is applied to user
        return 0.36  # Multiply chance of sharing by 0.36 times
    else:  # Otherwise
        return 1  # Does not affect sharing chance


class UserAgent(Agent):
    def __init__(
            self,
            unique_id,
            model,
            initial_state,
            followers,
            age,
            education,
            betweenness_centrality,
            closeness_centrality,
            degree_centrality,
            harmonic_centrality,
            relative_centrality,
            shared_from,
    ):
        super().__init__(unique_id, model)

        self.state = initial_state
        self.followers = followers
        self.age = age
        self.education = education
        self.betweenness_centrality = betweenness_centrality
        self.closeness_centrality = closeness_centrality
        self.degree_centrality = degree_centrality
        self.harmonic_centrality = harmonic_centrality
        self.relative_centrality = relative_centrality
        self.shared_from = shared_from

    def try_to_share_to_followers(self):
        neutral_followers = [  # Get a list of the node's neutral followers
            agent
            for agent in self.model.grid.get_cell_list_contents(self.followers)
            if agent.state is State.NEUTRAL
        ]

        for a in neutral_followers:  # For each neutral follower
            chance_to_share = self.model.share_chance * get_age_multiplier(a) * get_education_multiplier(a) * \
                              intervention(a)
            # Calculate the chance to share based on:
            # base share chance, the follower's age, the follower's education, and accuracy-nudge intervention
            if self.random.random() < chance_to_share:
                # Decide if the follower shares the information
                a.state = State.SHARED
                # Set its state to shared
                a.shared_from = self.unique_id
                # Set that the ID of the node which it shared from
            elif self.random.random() < self.model.refuse_chance:
                # If the agent did not share, decide if they refuse the information
                a.state = State.REFUSED
                # Sets its state to refused

    def step(self):
        if self.state is State.SHARED:  # For each agent that has shared the information
            self.try_to_share_to_followers()  # Try to share the information to its followers
