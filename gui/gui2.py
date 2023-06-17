import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from postprocess import PostProcess
from PIL import Image, ImageDraw, ImageFont
import os
cwd = os.getcwd()


class MainWindow:
    """
    The main window class for the Citation Evaluator application.
    """

    def __init__(self, root):
        """
        Initializes the MainWindow object.

        Args:
            root (tk.Tk): The root Tk object.
        """
        self.root = root
        self.root.title("Citation Evaluator")
        self.root.geometry("800x600")  # Set window size
        self.root.configure()  # Set background color
        self.graph_glob = None

        # Create a frame to hold the elements
        # frame = ttk.Frame(root)
        # frame.pack(pady=20)

        # Set background image
        # Replace with the actual path to your image
        image = Image.open(cwd+"\gui\main.png")
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(root, image=photo)
        label.image = photo
        label.place(relwidth=1, relheight=1)

        # Label
        label = ttk.Label(self.root, text="Citation Evaluator",
                          font=("Helvetica", 24), background="#DDDDDD")
        label.place(relx=0.35, rely=0.1, relheight=0.1)

        # Dropdown menu 1
        selected_option1 = tk.StringVar()
        self.dropdown1 = ttk.Combobox(
            self.root,
            textvariable=selected_option1,
            values=["Dataset", "Figshare", "Dryad", "Zenodo"],
            state="readonly",
            font=("Helvetica", 16),
            width=10,
        )
        self.dropdown1.current(1)
        self.dropdown1.place(relx=0.05, rely=0.5, relheight=0.08)

        # Dropdown menu 2
        selected_option2 = tk.StringVar()
        self.dropdown2 = ttk.Combobox(
            self.root,
            textvariable=selected_option2,
            values=["Time Interval", "Weeks", "Months", "Years"],
            state="readonly",
            font=("Helvetica", 16),
            width=10,
        )
        self.dropdown2.current(0)
        self.dropdown2.place(relx=0.25, rely=0.5, relheight=0.08)

        # Dropdown menu 3
        selected_option3 = tk.StringVar()
        self.dropdown3 = ttk.Combobox(
            self.root,
            textvariable=selected_option3,
            values=["Snapshot Number", "1", "3", "5"],
            state="readonly",
            font=("Helvetica", 16),
            width=15,
        )
        self.dropdown3.current(0)
        self.dropdown3.place(relx=0.45, rely=0.5, relheight=0.08)

        # Dropdown menu 4
        selected_option4 = tk.StringVar()
        values = ["Prediction percent", "5%", "15%", "30%"]
        self.dropdown4 = ttk.Combobox(
            self.root,
            textvariable=selected_option4,
            state="readonly",
            values=values,
            font=("Helvetica", 16),
            width=15,
        )
        self.dropdown4.current(0)
        self.dropdown4.place(relx=0.73, rely=0.5, relheight=0.08)

        style = ttk.Style()
        style.configure("TButton", font=(
            "Helvetica", 16))
        # Button
        button = ttk.Button(
            self.root,
            text="Predict",
            command=self.open_window2,
            width=25,
        )
        button.place(relx=0.43, rely=0.7, relheight=0.08, relwidth=0.2)

    def open_window2(self):
        """
        Opens the secondary window for graph processing.
        """
        self.root.withdraw()  # Hide the main window
        window2 = SecondaryWindow(
            self.root,
            2,
            "Graph Process",
            self.open_window1,
            self.open_window4,
            "Home",
            "Confusion Matrix",
            colors=True,
            graph1Btn=True,
            graph2Btn=True
        )
        window2.root.geometry("800x600")  # Set window size
        window2.root.configure(bg="lightgreen")  # Set background color
        self.set_window_position(window2.root)

    def open_window1(self):
        """
        Opens the primary window.
        """
        self.root.deiconify()  # Restore the main window if needed

    def open_window4(self):
        """
        Opens the secondary window for the confusion matrix.
        """
        self.root.withdraw()
        window4 = SecondaryWindow(
            self.root,
            4,
            "Confution Matrix",
            self.open_window1,
            self.open_window2,
            "Home",
            "Prediction",
        )
        window4.root.geometry("800x600")  # Set window size
        window4.root.configure(bg="lightpink")  # Set background color
        self.set_window_position(window4.root)

    def set_window_position(self, window):
        """
        Sets the position of the given window.

        Args:
            window: The window to set the position for.
        """
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (window.winfo_width() // 2)
        y = (window.winfo_screenheight() // 2) - (window.winfo_height() // 2)
        window.geometry("+{}+{}".format(x, y))


class SecondaryWindow:
    """
    The secondary window class for the Citation Evaluator application.
    """

    def __init__(
        self,
        root,
        numOfWindow,
        title,
        next_window1,
        next_window2,
        button_name1,
        button_name2,
        colors=False,
        graph1Btn=False,
        graph2Btn=False
    ):
        """
        Initializes the SecondaryWindow object.

        Args:
            root (tk.Toplevel): The root Toplevel object.
            numOfWindow (int): The number of the window.
            title (str): The title of the window.
            next_window1 (function): The function to open the next window 1.
            next_window2 (function): The function to open the next window 2.
            button_name1 (str): The name of button 1.
            button_name2 (str): The name of button 2.
            colors (bool, optional): Whether to display colors. Defaults to False.
            graph1Btn (bool, optional): Whether to display graph 1 button. Defaults to False.
            graph2Btn (bool, optional): Whether to display graph 2 button. Defaults to False.
        """
        self.root = tk.Toplevel(root)
        self.root.title(title)
        self.root.configure()  # Set background color

        self.next_window1 = next_window1
        self.next_window2 = next_window2

        # Set background image
        # Replace with the actual path to your image

        image = Image.open(cwd+"\gui\secWin.png")
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self.root, image=photo)
        label.image = photo
        label.place(x=0, y=0, relwidth=1, relheight=1)

        label = ttk.Label(self.root, text=title, font=(
            "Helvetica", 24), background="#DDDDDD")
        label.pack(pady=10)
        # post_process = PostProcess(
        # cwd+"\\results\link_prediction\CA-GrQctest_label.txt", cwd+"\data\link_prediction\CA-GrQc_test.txt", 1100, 1100)
        if numOfWindow == 4:
            # switch to a location based path

            label2 = ttk.Label(
                self.root, text="TP : {}".format(450), font=("Helvetica", 16))
            label2.place(relx=0.35, rely=0.2)
            label3 = ttk.Label(
                self.root, text="TN : {}".format(450), font=("Helvetica", 16))
            label3.place(relx=0.55, rely=0.2)
            label4 = ttk.Label(
                self.root, text="FP : {}".format(450), font=("Helvetica", 16))
            label4.place(relx=0.35, rely=0.3)
            label5 = ttk.Label(
                self.root, text="FN : {}".format(450), font=("Helvetica", 16))
            label5.place(relx=0.55, rely=0.3)
            label6 = ttk.Label(self.root, text="Accuracy : {:.3f}".format(
                0.525), font=("Helvetica", 16), background="#FFFFFF")
            label6.place(relx=0.15, rely=0.5)
            label7 = ttk.Label(self.root, text="Precision : {:.3f}".format(
                0.525), font=("Helvetica", 16), background="#FFFFFF")
            label7.place(relx=0.4, rely=0.5)
            label8 = ttk.Label(self.root, text="Recall : {:.3f}".format(
                0.525), font=("Helvetica", 16), background="#FFFFFF")
            label8.place(relx=0.65, rely=0.5)

        s = ttk.Style()
        s.configure('my.TButton', font=('Helvetica', 12, 'bold'))

        if graph1Btn:
            graphBtn1 = ttk.Button(
                self.root,
                text="show network",
                # command=lambda: show_graph(
                # cwd+'\data\link_prediction\CA-GrQc_train2.txt', post_process.unrecovered, post_process.recovered)
            )
            graphBtn1.place(relx=0.4, rely=0.6, relheight=0.08)

        # Create a frame to hold the elements
        if colors:
            # Create a red square and a label for negative neighbors
            frame = tk.Frame(self.root, background="#FFFFFF")
            frame.pack(padx=20, pady=20)
            frame.place(relx=0.35, rely=0.15, relheight=0.3, relwidth=0.3)
            red_square = tk.Label(frame, width=20, height=20,
                                  bg="red")
            red_square.place(relx=0.05, rely=0.05, relheight=0.2, relwidth=0.2)
            neg_label = tk.Label(
                frame, text="Negative Neighbors", background="#FFFFFF")
            neg_label.place(relx=0.2, rely=0.05, relheight=0.2)

            # Create a green square and a label for positive neighbors
            green_square = tk.Label(
                frame, width=20, height=20, bg="green")
            green_square.place(relx=0.05, rely=0.40,
                               relheight=0.2, relwidth=0.2)
            pos_label = tk.Label(
                frame, text="Positive Neighbors", background="#FFFFFF")
            pos_label.place(relx=0.2, rely=0.40, relheight=0.2)

            # Create a gray square and a label for original neighbors
            gray_square = tk.Label(
                frame, width=20, height=20, bg="gray")
            gray_square.place(relx=0.05, rely=0.75,
                              relheight=0.2, relwidth=0.2)
            org_label = tk.Label(
                frame, text="Original Neighbors", background="#FFFFFF")
            org_label.place(relx=0.2, rely=0.75, relheight=0.2)

        s2 = ttk.Style()
        s2.configure('my.TButton', font=('Helvetica', 12))
        button1 = ttk.Button(
            self.root,
            text=button_name1,
            command=lambda: self.switch_window(1)

        )
        button1.place(relx=0.2, rely=0.8, relheight=0.08)

        button2 = ttk.Button(
            self.root,
            text=button_name2,
            command=lambda: self.switch_window(2)
        )
        button2.place(relx=0.6, rely=0.8, relheight=0.08)

    def switch_window(self, num_of_window):
        """
        Switches to the specified window.

        Args:
            num_of_window (int): The number of the window.
        """
        self.root.destroy()

        if num_of_window == 1:
            self.next_window1()
        elif num_of_window == 2:
            self.next_window2()


def show_graph(file_path, edge_list_red, edge_list_green):
    """
    Show the graph based on the provided file path and edge lists.

    Args:
        file_path (str): The path to the file containing the graph data.
        edge_list_red (list): A list of edges to be displayed as red.
        edge_list_green (list): A list of edges to be displayed as green.
    """
    global G
    if G is None:
        G = create_graph(file_path, edge_list_red, edge_list_green)
    visualize_graph(G)


def create_graph(file_path, edge_list_red, edge_list_green):
    """
    Create a graph based on the provided file path and edge lists.

    Args:
        file_path (str): The path to the file containing the graph data.
        edge_list_red (list): A list of edges to be displayed as red.
        edge_list_green (list): A list of edges to be displayed as green.

    Returns:
        networkx.Graph: The created graph.
    """
    global G
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            u, v = line.strip().split()
            G.add_edge(u, v, color='gray')
    for u, v in edge_list_red:
        G.add_edge(u, v, color='red')
    for u, v in edge_list_green:
        G.add_edge(u, v, color='green')
    return G


def visualize_graph(graph):
    """
    Visualize the provided graph.

    Args:
        graph (networkx.Graph): The graph to visualize.
    """
    pos = nx.spring_layout(graph)
    edge_colors = [graph[u][v]['color'] for u, v in graph.edges()]
    nx.draw(graph, pos, with_labels=True, node_color='lightblue',
            edge_color=edge_colors, font_size=10, node_size=500, width=1.0, edge_cmap=plt.cm.get_cmap('Greys'))
    plt.show()


def main():
    """
    The main entry point of the program.
    """
    root = tk.Tk()
    main_window = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    global G
    G = None
    main()
