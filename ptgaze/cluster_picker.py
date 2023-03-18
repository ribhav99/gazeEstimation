from matplotlib import pyplot as plt
import numpy as np

class ClusterPicker:
    
    def __init__(self, x, y) -> None:
        self.count = 0
        self.clusters = []
        self.gaze_cluster_index = None
        self.circle = None
        self.fig, self.ax = fig, ax = plt.subplots()
        self.ax.scatter(x, y)
        self.gaze = -1
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        plt.show()

    
    def onclick(self, event):
        # Get the x and y coordinates of the mouse click
        x, y = event.xdata, event.ydata
        # Check if the click is within the axes limits
        if x is not None and y is not None:
            # Define the initial radius of the circle
            radius = 0.1

            # Create a circle patch with the specified centroid and radius
            self.circle = plt.Circle((x, y), radius, color='r', fill=False)

            # Add the circle patch to the plot
            self.ax.add_patch(self.circle)

            # Refresh the plot
            plt.draw()
    
    def onmove(self, event):
        # Check if the left mouse button is pressed and a circle patch exists
        if event.button == 1 and self.circle is not None:
            # Get the x and y coordinates of the mouse move
            x, y = event.xdata, event.ydata

            # Calculate the distance between the mouse click and mouse move
            radius = np.sqrt((x - self.circle.center[0])**2 + (y - self.circle.center[1])**2)

            # Update the radius of the circle
            self.circle.set_radius(radius)

            # Refresh the plot
            plt.draw()

    # Define a function to handle the mouse release event
    def onrelease(self, event):
        # Set the circle patch to None
        # output.append([self.circle.center[0], self.circle.center[1], speaker])
        radius = self.circle.radius
        self.circle.remove()
        if self.gaze == 1: # if it's a speaker
            new_circle = plt.Circle((self.circle.center[0], self.circle.center[1]), radius, color='g', fill=False)
            self.gaze_cluster_index = self.count
        else:
            new_circle = plt.Circle((self.circle.center[0], self.circle.center[1]), radius, color='b', fill=False)
        self.clusters.append(new_circle)
        self.count += 1
        self.ax.add_patch(new_circle)
        plt.draw()
        self.circle = None
    
    def on_press(self, event):
        if event.key == 'z':   
            self.onundo(event)
        if event.key == "h":
            self.gaze = self.gaze * -1


    def onundo(self, event):
        if self.clusters:
            circle = self.clusters.pop()
            self.count -= 1
            circle.remove()
            plt.draw()


if __name__ == "__main__":
    x = np.random.randint(0, 50, 50)
    y = np.random.randint(0, 50, 50)
    cluster_picker = ClusterPicker(x, y)
    print(cluster_picker.clusters[0].center)
