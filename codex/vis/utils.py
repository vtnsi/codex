import matplotlib.pyplot as plt
import textwrap

def set_plot_txt(title, title_size, xaxis, yaxis, axis_size):
    plt.gcf()

    plt.title(textwrap.fill(
                title,
                width=60
            ), 
            fontsize=title_size,
            pad=titlepad,
            weight="bold")

    


    return