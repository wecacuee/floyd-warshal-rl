""" Drop in replacement of opencv using matplotlib """
import time

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure

def is_cv2_matplotlib(cv2):
    if hasattr(cv2, 'am_i_matplotlib'):
        return cv2.am_i_matplotlib
    return False

class MPLAsCV(object):
    """
    MPL    -> cv2
    figure -> window
    axes   -> image (np.array) 
    """
    WINDOW_NORMAL = 'wn'
    def __init__(self):
        self.windows = dict()
        # In case want to check true nature
        # Strange name so that it unlikely to conflict with some cv2 attribute
        self.am_i_matplotlib = True
        self.fig_manager = dict()

    def white_img(self, imgsize, dpi = 100.0):
        fig = Figure(figsize = (imgsize[1], imgsize[0])
            , dpi = dpi)
        ax = fig.gca() if fig.axes else fig.add_axes([0, 0, 1, 1])
        ax.clear()
        ax.axis('equal')
        ax.set_xlim(0, imgsize[1])
        ax.set_ylim(0, imgsize[0])
        #ax.set_xticks([])
        #ax.set_yticks([])
        return ax

    def namedWindow(self, name, flags=None):
        if name not in self.windows:
            self.windows[name] = max(self.windows.values(), default=0) + 1

    def _mpl_color(self, color):
        color = list(reversed(color))
        return [c / 255. for c in color] if isinstance(color, (list, tuple))\
                else color

    def _mpl_coord(self, ax, xy):
        xy_dpi = xy / ax.figure.dpi
        _, h = ax.get_figure().get_size_inches()
        return [0, h] + xy_dpi * [1, -1]

    def _mpl_scale(self, ax, length):
        return length / ax.figure.dpi

    def rectangle(self, ax, x1, x2, color, thickness=1):
        color = self._mpl_color(color)
        x1 = self._mpl_coord(ax, x1)
        x2 = self._mpl_coord(ax, x2)
        ax.add_patch(
            mpl.patches.Rectangle((min(x1[0], x2[0]), min(x1[1], x2[1]))
                                  , abs(x2[0]-x1[0])
                                  , abs(x2[1]-x1[1])
                                  , edgecolor=color
                                  , facecolor=color if thickness < 0 else 'w'
                                  , linewidth=thickness))

    def circle(self, ax, center, radius, color, thickness=1):
        color = self._mpl_color(color)
        center = self._mpl_coord(ax, center)
        radius = self._mpl_scale(ax, radius)
        ax.add_patch(
            mpl.patches.Circle(center
                               , radius
                               , edgecolor=color
                               , facecolor=color if thickness < 0 else 'w'
                               , linewidth=thickness))

    def line(self, ax, x1, x2, color, thickness=1):
        color = self._mpl_color(color)
        x1, x2 = (self._mpl_color(ax, x) for x in (x1, x2))
        ax.add_artist(
            mpl.lines.Line2D([x1[0], x2[0]],
                             [x1[1], x2[1]],
                             color=color, linewidth=thickness))

    def arrowedLine(self, ax, pt1, pt2, color, thickness=1, tipLength=0.1):
        pt1 = self._mpl_coord(ax, pt1)
        pt2 = self._mpl_coord(ax, pt2)
        color = self._mpl_color(color)
        delta = pt2 - pt1
        ax.arrow(pt1[0], pt2[1],
                 delta[0], delta[1],
                 width = thickness / ax.figure.dpi,
                 head_length = tipLength)

    def polylines(self, ax, pts, isClosed , color, thickness=1):
        color = self._mpl_color(color)
        pts = self._mpl_coord(ax, pts)
        ax.add_patch(
            mpl.patches.Polygon(np.asarray(pts)
                                , linewidth=thickness))

    def fillConvexPoly(self, ax, pts, color):
        color = self._mpl_color(color)
        pts = self._mpl_coord(ax, pts)
        ax.add_patch(
            mpl.patches.Polygon(np.asarray(pts) , facecolor=color
                               , linewidth=0))

    def putText(self, ax, text, xy, org, fontFace, fontScale, color):
        color = self._mpl_color(color)
        xy, org = (self._mpl_color(ax, x) for x in (xy, org))
        ax.add_patch(
            mpl.patches.Text(x      = xy[0],
                             y      = xy[1],
                             text   = text,
                             color  = color,
                             family = fontFace,
                             size   = fontScale))

    def imshow(self, name, ax):
        self.namedWindow(name)
        from matplotlib.backends.backend_tkagg import new_figure_manager_given_figure
        if name not in self.fig_manager:
            self.fig_manager[name] = new_figure_manager_given_figure(
                self.windows[name], ax.get_figure())
        else:
            self.fig_manager[name].canvas.figure = ax.get_figure()
            ax.get_figure().canvas = self.fig_manager[name].canvas

        self.fig_manager[name].canvas.draw()
        self.fig_manager[name].show()


    def waitKey(self, milliseconds):
        from matplotlib.backends.backend_tkagg import Tk
        if milliseconds <= 0:
            for manager in self.fig_manager.values():
                manager.show()
            Tk.mainloop()
        else:
            time.sleep(milliseconds/1000.)

    def imwrite(self, filename, ax):
        fig = plt.gcf()
        fig.sca(ax)
        plt.savefig(filename)

    def destroyWindow(self, name):
        self.fig_manager[name].destroy()
        del self.fig_manager[name]

    def destroyAllWindows(self):
        for manager in self.fig_manager.values():
            manager.destroy()

    def from_ndarray(self, arr):
        ax = white_img(arr.shape[:2])
        ax.imshow(arr)
        return ax

    def to_ndarray(self, ax):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(ax.get_figure())
        w, h = canvas.get_width_height()
        return np.frombuffer(canvas.tostring_argb(), dtype='u1').reshape(w, h, 4)

    def __del__(self):
        self.destroyAllWindows()

if __name__ == '__main__':
    WAIT_msecs = 1000
    import cv2
    actual_opencv = cv2
    for USE_MATPLOTLIB in (0, 1):
        if USE_MATPLOTLIB:
            cv2 = MPLAsCV()
        else:
            cv2 = actual_opencv
            cv2.white_img = lambda imsize : np.ones(
                (imsize[0], imsize[1], 3))*255.

        img = cv2.white_img((100, 300))
        cv2.rectangle(img, (10, 10), (40, 40), color=(255, 0, 0), thickness=4)
        cv2.imshow("c", img)
        cv2.waitKey(WAIT_msecs)
        cv2.circle(img, (60, 20), 10, color=(0, 255, 0), thickness=4)
        cv2.imshow("c", img)
        cv2.waitKey(WAIT_msecs)
        cv2.line(img, (30, 30), (60, 20), color=(0, 0, 255), thickness=4)
        cv2.imshow("c", img)
        cv2.waitKey(WAIT_msecs)
        cv2.fillConvexPoly(img
                           , np.int32([  [40, 60]
                                       , [40, 40]
                                       , [60, 40]
                                       , [60, 60]])
                           , color=(0, 255, 255))
        cv2.imshow("c", img)
        cv2.waitKey(WAIT_msecs)
        cv2.imwrite("/tmp/test%d.png" % USE_MATPLOTLIB, img)
