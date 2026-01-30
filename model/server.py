import math
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import NetworkModule, BarChartModule
from mesa.visualization.modules import TextElement
from mesa.visualization.modules.ChartVisualization import ChartModule
from .model import SocialMediaNetwork, State, number_shared, number_refused, number_neutral, centrality_colour


def network_portrayal(graph):
    def node_color(agent):
        return {State.SHARED: "#e71837", State.REFUSED: "#006400"}.get(
            # #e71837 is red, #006400 is green
            agent.state, centrality_colour(agent)
            # if agent is not SHARED or REFUSED use centrality_colour
        )

    def edge_color(agent1, agent2):
        if State.SHARED in (agent1.state, agent2.state) and agent1.state == agent2.state:
            # If both agents are SHARED:
            return "#f59aa8"  # #f59aa8 is light red
        elif (State.SHARED, State.REFUSED) == (agent1.state, agent2.state) \
                or (State.REFUSED, State.SHARED) == (agent1.state, agent2.state):
            # If, in a pair, one agent is SHARED and one is REFUSED:
            return "#90EE90"  # #90EE90 is light green
        return "#e8e8e8"  # #e8e8e8 is grey

    def edge_width(agent1, agent2):
        if State.REFUSED in (agent1.state, agent2.state):
            # If either of the nodes are REFUSED
            return 3
        return 2

    def get_agents(source, target):
        # Gets two agents based on their IDs
        return graph.nodes[source]["agent"][0], graph.nodes[target]["agent"][0]

    portrayal = dict()
    # Create an empty dictionary
    portrayal["nodes"] = [
        # For each node:
        {
            "size": 6,  # Node size is 6
            "color": node_color(agents[0]),  # Sets the colour of the node based on the node_color function
            "tooltip": f"Agent ID: {agents[0].unique_id}<br>State: {agents[0].state.name}"
                       f"<br>Age: {agents[0].age}<br>Education: {agents[0].education}"
                       f"<br>Followers ({len(agents[0].followers)}):"
                       f"<br> {' '.join([(str(i) + ' ') for i in agents[0].followers])}"
                       f"<br>Betweenness Centrality Score: {agents[0].betweenness_centrality}"
                       f"<br>Closeness Centrality Score: {agents[0].closeness_centrality}"
                       f"<br>Degree Centrality Score: {agents[0].degree_centrality}"
                       f"<br>Harmonic Centrality Score: {agents[0].harmonic_centrality}"
                       f"<br>{'Shared from %i' % agents[0].shared_from if agents[0].shared_from is not None else ''}",
            # This sets a tooltip which appears when hovering over a node. It contains a node's:
            # ID, age, number of followers, list of followers, all centrality scores,
            # and, if it is SHARED, which node it shared the information from
        }
        for (_, agents) in graph.nodes.data("agent")
    ]

    directed = graph.nodes.data("agent")[1][0].model.directed
    # Check if the network is directed based on the user set parameter

    portrayal["edges"] = [
        # For each edge:
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            # Calculate colour based on the state of both agents
            "width": edge_width(*get_agents(source, target)),
            # Calculate colour based on the state of both agents
            "directed": directed,
            # Set the edge to be directed or not
            "marker_size": "large",
            # Arrow size for directed edges is set to large
        }
        for (source, target) in graph.edges
    ]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500)
# Create a new network_portrayal
# Create a network graph visualisation based on network_portrayal
# with width and height of 500 pixels

state_chart = ChartModule(
    # Create a line chart for the number of nodes in each state
    series=[
        {"Label": "Shared", "Color": "#FF0000"},
        {"Label": "Neutral", "Color": "#808080"},
        {"Label": "Refused", "Color": "#008000"},
    ],
    canvas_width=600,
    # Chart has width of 600 pixels
)

age_chart = BarChartModule(
    # Create a bar chart for the number of nodes in each age group
    fields=[
        {"Label": "Aged 13-17", "Color": "#FF8A8A"},
        {"Label": "Aged 18-24", "Color": "#FF2400"},
        {"Label": "Aged 25-34", "Color": "#A30000"},
        {"Label": "Aged 35-49", "Color": "#800000"},
        {"Label": "Aged 50+", "Color": "#750000"},
    ],
    scope="model",
    sorting="none",
    canvas_width=750,
    # Based on data from the model, with no sorting, and a width of 750 pixels
)

education_chart = BarChartModule(
    # Create a bar chart for the number of nodes in each education level
    fields=[
        {"Label": "No diploma", "Color": "#FF8A8A"},
        {"Label": "Secondary diploma", "Color": "#FF2400"},
        {"Label": "Post-secondary diploma", "Color": "#A30000"},
    ],
    scope="model",
    sorting="none",
    canvas_width=750,
    # Based on data from the model, with no sorting, and a width of 750 pixels
)


class MyTextElement(TextElement):
    # Create a text element to be displayed under the graph
    def render(self, model):
        neutral_text = str(number_neutral(model))  # Number of neutral nodes
        shared_text = str(number_shared(model))  # Number of shared nodes
        refused_text = str(number_refused(model))  # Number of refused nodes
        shared_neutral_ratio = model.shared_neutral_ratio()
        shared_neutral_ratio_text = "&infin;" if shared_neutral_ratio is math.inf else f"{shared_neutral_ratio:.2f}"
        refused_neutral_ratio = model.refused_neutral_ratio()
        refused_neutral_ratio_text = "&infin;" if refused_neutral_ratio is math.inf else f"{refused_neutral_ratio:.2f}"

        return f"Neutral nodes: {neutral_text}<br>" \
               f"Nodes who have shared: {shared_text}<br>" \
               f"Nodes who have refused: {refused_text}<br>" \
               f"Shared/Neutral Ratio: {shared_neutral_ratio_text}<br>" \
               f"Refused/Neutral Ratio: {refused_neutral_ratio_text}"
        # Displays the text


model_params = {
    "num_nodes": UserSettableParameter(
        param_type="slider",
        name="Number of Agents",
        value=100,
        min_value=50,
        max_value=300,
        step=1,
        description="Number of agents to include in the OSNModel.",
    ),
    "num_nodes_desc": UserSettableParameter(
        param_type="static_text",
        value="Number of Agents is the number of nodes, representing agents, which will appear on the network "
              "graph. The default value of 100 agents is recommended. Larger network take considerably longer to "
              "simulate.",
    ),
    "share_chance": UserSettableParameter(
        param_type="slider",
        name="Share Chance (%)",
        value=50,
        min_value=1,
        max_value=100,
        step=1,
        description="Chance for the information to be shared to an agent's followers.",
    ),
    "refuse_chance": UserSettableParameter(
        param_type="slider",
        name="Refusal Chance (%)",
        value=100,
        min_value=0,
        max_value=100,
        step=1,
        description="Chance for the information to be refused by an agent's followers if it is not shared.",
    ),
    "chance_desc": UserSettableParameter(
        param_type="static_text",
        value="Share Chance is the base percentage chance that information will be shared from a node which has "
              "already shared information to a neutral node. Refusal Chance is the base percentage chance that a "
              "neutral node will refuse to share information if it chooses not to share it.",
    ),
    "num_initial_nodes": UserSettableParameter(
        param_type="slider",
        name="Initial Shares",
        value=1,
        min_value=1,
        max_value=10,
        step=1,
        description="Initial number of agents who shared the information.",
    ),
    "initial_nodes_high_degree": UserSettableParameter(
        param_type="checkbox",
        name="High Degree Initial Shares",
        value=True,
        description="Whether the initial agents who shared the information have a high node degree on the network.",
    ),
    "initial_desc": UserSettableParameter(
        param_type="static_text",
        value="Initial Shares is the number of nodes which will have initially shared information in the network. "
              "High Degree Initial Shares denotes whether the nodes which will have initially shared information in "
              "the network have high degree (i.e. a high number of connections).",
    ),
    "directed": UserSettableParameter(
        param_type="checkbox",
        name="Directed Graph",
        value=False,
        description="Whether the edges between nodes are directed or not.",
    ),
    "dir_desc": UserSettableParameter(
        param_type="static_text",
        value="Directed Graph denotes whether the edges between nodes are directed or not.",
    ),
    "centrality_view": UserSettableParameter(
        param_type="choice",
        name="Centrality Colour View",
        value='Degree',
        choices=['Betweenness', 'Closeness', 'Degree', 'Harmonic'],
        description="Type of centrality to be used when colouring the nodes on the graph.",
    ),
    "cen_desc": UserSettableParameter(
        param_type="static_text",
        value="Centrality Colour View denotes the centrality score type which will be considered when colouring the "
              "neutral nodes. For each node, a higher centrality score (of the selected type) will give the node a "
              "darker shade of blue as its colour. The centrality score for each type of node will be shown in the "
              "node's hover information. ",
    ),
    "graph_type": UserSettableParameter(
        param_type="choice",
        name="Graph Model",
        value='Degree Sequence',
        choices=['Degree Sequence', 'Erdős–Rényi', 'Barabási–Albert', 'Completely Connected'],
        description="Graph OSNModel to be generated for the network.",
    ),
    "power_law": UserSettableParameter(
        param_type="slider",
        name="Power Law Value (For Degree Sequence)",
        value=2.5,
        min_value=2.0,
        max_value=5.0,
        step=0.1,
        description="Power law value to be used when generating a degree sequence OSNModel.",
    ),
    "erdos_renyi_edge_probability": UserSettableParameter(
        param_type="slider",
        name="Edge Probability (For Erdős–Rényi)",
        value=0.05,
        min_value=0.01,
        max_value=1,
        step=0.01,
        description="Edge probability to be used when generating an Erdős–Rényi OSNModel.",
    ),
    "barabasi_albert_num_edges": UserSettableParameter(
        param_type="slider",
        name="Number of Edges per Node (For Barabási–Albert)",
        value=3,
        min_value=1,
        max_value=10,
        step=1,
        description="Number of edges per node to be used when generating a Barabási–Albert OSNModel.",
    ),
    "type_desc": UserSettableParameter(
        param_type="static_text",
        value="Graph Model denotes the type of OSNModel which will be generated for the network graph. Power Law Value,"
              "Edge Probability, and Number of Edges per Node are used in calculations to generate the network graph, "
              "depending on the chosen OSNModel. More information is available in the About section.",
    ),
    "accuracy_nudge_intervention": UserSettableParameter(
        param_type="slider",
        name="Accuracy-Nudge Intervention Chance (%)",
        value=0,
        min_value=0,
        max_value=100,
        step=1,
        description="The percentage chance that an accuracy-nudge intervention will be applied to an agent.",
    ),
    "intervention_desc": UserSettableParameter(
        param_type="static_text",
        value="Accuracy-Nudge Intervention Chance is the percentage chance that an accuracy-nudge intervention will "
              "be applied to a node when attempting to share information. The intervention, when applied, reduces the "
              "chance of sharing information by 50%. More information is available in the About Section.",
    ),
    "perc_13_17": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents Aged 13-17",
        value=7,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents who are within the age range 13-17.",
    ),
    "perc_18_24": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents Aged 18-24",
        value=17,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents who are within the age range 18-24.",
    ),
    "perc_25_34": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents Aged 25-34",
        value=38,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents who are within the age range 25-34.",
    ),
    "perc_35_49": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents Aged 35-49",
        value=21,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents who are within the age range 35-49.",
    ),
    "perc_50": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents Aged 50+",
        value=17,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents who are aged 50 and over.",
    ),
    "age_desc": UserSettableParameter(
        param_type="static_text",
        value="The above sliders denote how many nodes in the network will be assigned each age group (e.g. if "
              "Percentage of Users Aged 35-49 is set to 21 then 21% of the nodes in the network will be assigned an "
              "age value between 35-49). A node's age affects the chance that it will share information. More "
              "information is available in the About section. ",
    ),
    "perc_no_diploma": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents with No Diploma",
        value=12,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents with no diploma.",
    ),
    "perc_secondary_diploma": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents with Secondary Diploma",
        value=42,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents with a secondary diploma.",
    ),
    "perc_post_secondary_diploma": UserSettableParameter(
        param_type="slider",
        name="Percentage of Agents with Post-Secondary Diploma",
        value=46,
        min_value=0,
        max_value=100,
        step=1,
        description="Percentage of agents with a post-secondary diploma.",
    ),
    "ed_desc": UserSettableParameter(
        param_type="static_text",
        value="The above sliders denote how many nodes in the network will be assigned each education level (e.g. if "
              "Percentage of Users with Secondary Diploma is set to 42 then 42% of the nodes in the network will be "
              "assigned an education level of Secondary diploma. A node's education affects the chance that it will "
              "share information. More information is available in the About section. ",
    ),
}

server = ModularServer(
    # Create a new visualisation with the following elements:
    SocialMediaNetwork,  # Network model to create
    [network, MyTextElement(), state_chart, age_chart, education_chart],  # Visualisation elements
    "OSN Model",  # Title
    model_params  # User settable parameters
)
server.port = 8522
# Set the server port ID for the web interface to 8552
