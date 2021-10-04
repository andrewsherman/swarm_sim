from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from matplotlib.transforms import Affine2D


class TriBot(Polygon):
    def __init__(self, xy, radius, **kwargs):
        vert = []
        for theta in np.linspace(0, 2*np.pi, 4):
            vert.append(radius * np.array([np.cos(theta), np.sin(theta)]))
        vert = np.array([*vert[:2], np.zeros(2), *vert[2:]])
        self.radius = radius
        self.centre = np.array(xy)
        self.colour = np.random.rand(3)
        self.colour /= np.linalg.norm(self.colour)
        super().__init__(vert, **kwargs, fc=self.colour, ec='black')
        self.velocity = np.random.randn(2) * 10
        self.velocity = 10 * self.velocity/np.linalg.norm(self.velocity)
        self.perceptive_field = Point(xy).buffer(radius*50)

    def perceives(self, other):
        if self.perceptive_field.contains(Point(other.centre)):
            return True

    def move(self, dt, world):
        assert self in world.ax.patches

        np.random.shuffle(world.patches)
        for other in [o for o in world.patches if o is not self]:
            if self.perceives(other):
                # keep away
                vec = self.centre - other.centre
                norm = np.linalg.norm(vec)
                # if (10 * (vec / norm) / norm ** 2 > 1).any():
                #     print(10 * (vec / norm) / norm**2)
                D = self.radius * 5
                # force = np.exp(-D*norm)*D*norm - norm**2/(np.exp(D))
                force = np.exp(-D * norm) - norm ** 2 / (np.exp(D))
                self.velocity += dt * force * vec / norm
                norm = np.linalg.norm(self.velocity)
                if norm > 10:
                    self.velocity = 10 * self.velocity / norm
                # try to align
                norm = np.linalg.norm(self.velocity)
                if norm > 15:
                    self.velocity = 15 * self.velocity / norm
                    norm = 15
                self.velocity += 2 * other.velocity
                self.velocity *= norm / np.linalg.norm(self.velocity)

        # find time to intersect with all walls
        t = 0
        while t < dt:
            left = (self.radius-self.centre[0])/self.velocity[0]
            right = (world.shape[0]-self.centre[0]-self.radius)/self.velocity[0]
            bottom = (self.radius-self.centre[1])/self.velocity[1]
            top = (world.shape[1]-self.centre[1]-self.radius)/self.velocity[1]
            times = [left, right, bottom, top]
            valid_indexes = [i for i, s in enumerate(times) if 0 < s <= dt-t]
            if not valid_indexes:
                self.centre = self.centre + (dt-t) * self.velocity
                self.centre[0] = np.clip(self.centre[0], self.radius, world.shape[0] - self.radius)
                self.centre[1] = np.clip(self.centre[1], self.radius, world.shape[1] - self.radius)
                break
            times = [times[i] for i in valid_indexes]
            i = valid_indexes[times.index(min(times))]

            # plt.plot([self.centre[0], self.centre[0]+min(times)*self.velocity[0]],
            #          [self.centre[1], self.centre[1]+min(times)*self.velocity[1]], self.colour)
            self.centre = self.centre + min(times) * self.velocity
            self.centre[0] = np.clip(self.centre[0], self.radius, world.shape[0]-self.radius)
            self.centre[1] = np.clip(self.centre[1], self.radius, world.shape[1] - self.radius)
            t += min(times)
            # now update v
            if i == 0 or i == 1:
                # collided with left or right wall
                self.velocity[0] *= -1
            else:
                # collided with top or bottom wall
                self.velocity[1] *= -1
        theta = np.arctan2(self.velocity[1], self.velocity[0])
        trans = Affine2D()
        trans.rotate(theta)
        trans.translate(*self.centre)
        self.set_transform(trans)
        if not (0 < self.centre[0] < world.shape[0]) or not (0 < self.centre[1] < world.shape[1]):
            world.remove(self)
        self.perceptive_field = Point(self.centre).buffer(radius * 50)
        # self.set_transform()
        # self.draw()


class World:
    def __init__(self, shape):
        # set up the matplotlib elements
        self.shape = shape
        self.fig = plt.figure('Swarm World')
        dpi = self.fig.dpi
        self.fig.set_size_inches(shape[0]/dpi, shape[1]/dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.set_xlim([0, shape[0]])
        self.ax.set_ylim([0, shape[1]])
        plt.grid('on')

        # initialise patch list
        self.patches = []

    def update(self, dt):
        for p in self.patches:
            p.move(dt, self)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def add(self, patch):
        self.patches += [patch]
        self.ax.add_patch(patch)

    def remove(self, patch):
        self.patches.pop(self.patches.index(patch))
        patch.remove()


if __name__=="__main__":
    plt.ion()
    w = World((1000, 800))
    N=40
    radius = 5
    p = [TriBot((np.random.randint(0+radius, w.shape[0]-radius), np.random.randint(0+radius, w.shape[1]-radius)), radius) for _ in range(N)]
    import time
    t0 = time.time()
    for p_ in p:
        w.add(p_)
    plt.plot([radius, w.shape[0]-radius, w.shape[0]-radius, radius, radius],
             [w.shape[1]-radius, w.shape[1]-radius, radius, radius, w.shape[1]-radius])
    # print(p.velocity)
    i=0
    while time.time() - t0 < 100:
        i+=1
        if not w.ax.texts:
            plt.text(w.shape[0]/2, w.shape[1]-20, f'N tribots = {len(w.ax.patches)}')
        else:
            w.ax.texts[0] = plt.text(w.shape[0]/2, w.shape[1]-20, f'N tribots = {len(w.ax.patches)}')
        w.update(1)
        if len(w.ax.lines) > 10:
            w.ax.lines = [w.ax.lines[0], *w.ax.lines[-10:]]
