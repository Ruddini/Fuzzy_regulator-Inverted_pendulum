import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as ani
import numpy as np
from numpy import sin, cos, arctan2, pi
from itertools import cycle
from sys import argv, exit

class InvertedPendulum():
    '''Inicjalizacja stałych:
    M - masa wózka
    m - masa kulki
    l - długość ramienia wahadła

    Warunków początkowych:
    x0 - początkowe położenie wózka
    dx0 - początkowa prędkość wózka
    theta0 - początkowe położenie wahadla
    dtheta0 - początkowa prędkość wahadła

    Zakłócenia zewnętrznego:
    dis_cyc - zmienna odpowiada za to, czy zakłócenie jest zapętlone
    disruption - wartości zakłócenia w kolejnych chwilach czasowych

    Parametry planszy/obrazka:
    iw, ih - szerokość i wysokość obrazka
    x_max - maksymalna współrzędna pozioma (oś x jest symetryczna, więc minimalna wynosi -x_max)
    h_min - minialna współrzędna pionowa
    h_max - maksymalna współrzędna pionowa

    Powyższe dane są pobierane z pliku jeśli zmienna f_name nie jest pusta'''
    def __init__(self, M=10, m=5, l=50, x0=0, theta0=0, dx0=0, dtheta0=0, dis_cyc=True, disruption=[0], iw=1000, ih=500, x_max=100, h_min=0, h_max=100, f_name=None):
        if f_name:
            with open(f_name) as f_handle:
                lines = f_handle.readlines()
                init_cond = lines[0].split(' ')
                self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = [float(el) for el in init_cond[:7]]
                self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = [int(el) for el in init_cond[-5:]]
                if lines[1]:
                    self.disruption = cycle([float(el) for el in lines[2].split(' ')])
                else:
                    self.disruption = iter([float(el) for el in lines[2].split(' ')])
        else:
            self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = M, m, l, x0, theta0, dx0, dtheta0
            self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = iw, ih, x_max, h_min, h_max
            if dis_cyc:
                self.disruption = cycle(disruption)
            else:
                self.disruption = iter(disruption)

    # Funkcja odpowiedzialna za wyłączenie programu, gdy zostanie zamknięte okno z rysunkami
    def handle_close(self, evt):
        exit(0)

    # Inicjalizacja obrazka
    def init_image(self, x, theta):
        dpi = 100
        self.fig, ax = plt.subplots(figsize=(self.image_w/dpi, self.image_h/dpi), dpi=dpi)
        plt.autoscale(False)
        plt.xticks(range(-self.x_max, self.x_max+1, int(self.x_max/10)))
        plt.yticks(range(self.h_min, self.h_max+1, int(self.h_max/10)))
        self.hor = 10
        self.c_w = 16
        self.c_h = 8
        r = 4
        self.cart = patches.Rectangle((x-self.c_w/2, self.hor-self.c_h/2), self.c_w, self.c_h,linewidth=1, edgecolor='blue',facecolor='blue', zorder=2)
        self.blob = patches.Circle((x-self.l*sin(theta), self.hor+self.l*cos(theta)), r ,linewidth=1, edgecolor='red',facecolor='red', zorder=3)
        self.guide = patches.Rectangle((-self.x_max, self.hor-1), 2*self.x_max, 2, edgecolor='black',facecolor='black', zorder=0)
        self.arm = patches.Rectangle((x+cos(theta), self.hor+sin(theta)), self.l, 2, 180*theta/np.pi+90, edgecolor='brown', facecolor='brown', zorder=1)
        ax.add_patch(self.cart)
        ax.add_patch(self.blob)
        ax.add_patch(self.guide)
        ax.add_patch(self.arm)

    #    Aktualizacja położenia wózka, ramienia i wahadła
    def update_image(self, data):
        x, theta = data[0], data[1]
        self.cart.set_x(x-self.c_w/2)
        self.blob.set_center((x-self.l*sin(theta), self.hor+self.l*cos(theta)))
        self.arm.angle = theta*180/np.pi+90
        self.arm.set_xy((x+cos(theta), self.hor+sin(theta)))
        return self.cart, self.blob, self.arm,

    # Rozwiązanie równań mechaniki wahadła
    def solve_equation(self, x, theta, dx, dtheta, F):
        l, m, M = self.l, self.m, self.M
        g = 9.81
        a11 = M+m
        a12 = -m*l*cos(theta)
        b1 = F-m*l*dtheta**2*sin(theta)
        a21 = -cos(theta)
        a22 = l
        b2 = g*sin(theta)
        a = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        sol = np.linalg.solve(a, b)
        return sol[0], sol[1]

    # Scałkowanie numeryczne przyśpieszenia, żeby uzyskać pozostałe parametry układu
    def count_state_params(self, x, theta, dx, dtheta, F, dt=0.001):
        ddx, ddtheta = self.solve_equation(x, theta, dx, dtheta, F)
        dx += ddx*dt
        x += dx*dt
        dtheta += ddtheta*dt
        theta += dtheta*dt
        theta = arctan2(sin(theta), cos(theta))
        return x, theta, dx, dtheta

    # Funkcja generująca kolejne dane symulacji
    def generate_data(self):
        x = self.x0
        theta = self.theta0
        dx = self.dx0
        dtheta = self.dtheta0
        while True:
            for i in range(self.frameskip+1):
                dis=next(self.disruption, 0)
                control = self.fuzzy_control(x, theta, dx, dtheta)
                F = dis+control
                x, theta, dx, dtheta = self.count_state_params(x, theta, dx, dtheta, F)
                if not self.sandbox:
                    if x < -self.x_max or x > self.x_max or np.abs(theta) > np.pi/3:
                        exit(1)
            yield x, theta

    # Uruchomienie symulacji
    # Zmienna sandbox mówi o tym, czy symulacja ma zostać przerwana w przypadku nieudanego sterowania -
    # - to znaczy takiego, które pozwoliło na zbyt duże wychylenia iksa lub na zbyt poziomo położenie wahadła
    def run(self, sandbox, frameskip=200):
        self.init_image(self.x0, self.theta0)
        self.sandbox = sandbox
        self.frameskip = frameskip
        a = ani.FuncAnimation(self.fig, self.update_image, self.generate_data, interval=1)
        plt.show()

    # Regulator rozmyty, który trzeba zaimplementować
    def fuzzy_control(self, x, theta, dx, dtheta):
        if (x or theta or dx or dtheta):
            fuzyfication = self.fuzzyfication(x,theta,dx,dtheta)
            roles = self.fuzzy_rules(fuzyfication)
            force = self.defuzyfication(roles)
            print(f"x: {x}, theta: {theta}, dx: {dx}, dtheta: {dtheta}")
            return force
        else: return 0


    def trapezoid(self, x, start, start1, end1, end):
        # jako argumenty należy podać wartości x wierzchołków trapezu
        # zwraca wartość x dla funckji trapezu
        self.x = x
        if start <= self.x and self.x < start1:
            funcx = (self.x - start) / (start1 - start)
        elif start1 <= self.x and self.x <= end1:
            funcx = 1
        elif end1 < self.x and self.x <= end:
            funcx = (end - self.x) / (end - end1)
        else:
            return False
        return funcx

    def fuzzyfication(self, x, theta, dx, dtheta):

        """
        Położenie X
          DLX       BLX     ZX    BPX        DPX
        _______    ______        ______    _______
        |      \  /      \  /\  /      \  /       |
        |       \/        \/  \/        \/        |
        |       /\        /\  /\        /\        |
        |      /  \      /  \/  \      /  \       |

        Położenie theta
        DLtheta   BLtheta Ztheta BPtheta   DPtheta
        _______    ______        ______    _______
        |      \  /      \  /\  /      \  /       |
        |       \/        \/  \/        \/        |
        |       /\        /\  /\        /\        |
        |      /  \      /  \/  \      /  \       |

        Prędkość liniowa wózka
          WLX       MLX     ZDX    MPX        WPX
        _______    ______        ______    _______
        |      \  /      \  /\  /      \  /       |
        |       \/        \/  \/        \/        |
        |       /\        /\  /\        /\        |
        |      /  \      /  \/  \      /  \       |

        Prędkość kątowa wachadła
        WLtheta   MLtheta ZDtheta MPtheta   WPtheta
        _______    ______        ______    _______
        |      \  /      \  /\  /      \  /       |
        |       \/        \/  \/        \/        |
        |       /\        /\  /\        /\        |
        |      /  \      /  \/  \      /  \       |
        """

        result_set = {
            "DLX": 0, "BLX": 0,"ZX":0, "BPX": 0, "DPX": 0,
            "DLtheta": 0, "BLtheta": 0, "Ztheta":0, "BPtheta": 0, "DPtheta": 0,
            "WLX": 0, "MLX": 0, "ZDX": 0, "MPX": 0, "WPX": 0,
            "WLtheta": 0, "MLtheta": 0, "ZDtheta": 0, "MPtheta": 0, "WPtheta": 0
        }
        """1. Okrślenie rozmywania wartości położenia wózka"""
        # podzieliłem położenie wózka na 5 przedziałów trapezowych
        side_rang_x = 6
        x_range = [-150,- 100,0, 100, 150]
        DLX_range = [x_range[0], x_range[1] + side_rang_x / 2]
        BLX_range = [x_range[1] - side_rang_x / 2, x_range[2]]
        ZX_range = [-side_rang_x,side_rang_x]
        BPX_range = [x_range[2], x_range[3] + side_rang_x / 2]
        DPX_range = [x_range[3] - side_rang_x / 2, x_range[4]]

        if DLX_range[0] < x and x < DLX_range[1]:
            DLX = self.trapezoid(x, DLX_range[0], DLX_range[0], DLX_range[1] - side_rang_x, DLX_range[1])
            result_set["DLX"] = DLX
        if BLX_range[0] < x and x < BLX_range[1]:
            BLX = self.trapezoid(x, BLX_range[0], BLX_range[0] + side_rang_x, BLX_range[1] - side_rang_x, BLX_range[1])
            result_set["BLX"] = BLX
        if ZX_range[0]<x and x<ZX_range[1]:
            ZX = self.trapezoid(x, ZX_range[0], ZX_range[0] + side_rang_x, ZX_range[1] - side_rang_x, ZX_range[1])
            result_set["ZX"] = ZX
        if BPX_range[0] < x and x < BPX_range[1]:
            BPX = self.trapezoid(x, BPX_range[0], BPX_range[0] + side_rang_x, BPX_range[1] - side_rang_x, BPX_range[1])
            result_set["BPX"] = BPX
        if DPX_range[0] < x and x < DPX_range[1]:
            DPX = self.trapezoid(x, DPX_range[0], DPX_range[0] + side_rang_x, DPX_range[1], DPX_range[1])
            result_set["DPX"] = DPX

        """2. Okrślenie rozmywania wartości położenia kątowego wachadła"""
        # podzieliłem położenie wachadła na 5 przedziałów trapezowych
        side_rang_theta = sin(pi / 36)
        theta_range = [-5, -sin(pi / 12), 0, sin(pi / 12), 5]
        DPtheta_range = [theta_range[0], theta_range[1] + side_rang_theta / 2]
        BPtheta_range = [theta_range[1] - side_rang_theta / 2, theta_range[2]]
        Ztheta_range = [-side_rang_theta,side_rang_theta]
        BLtheta_range = [theta_range[2], theta_range[3] + side_rang_theta / 2]
        DLtheta_range = [theta_range[3] - side_rang_theta / 2, theta_range[4]]

        if DLtheta_range[0] < theta and theta < DLtheta_range[1]:
            DLtheta = self.trapezoid(theta, DLtheta_range[0], DLtheta_range[0], DLtheta_range[1] - side_rang_theta,
                                     DLtheta_range[1])
            result_set["DLtheta"] = DLtheta
        if BLtheta_range[0] < theta and theta < BLtheta_range[1]:
            BLtheta = self.trapezoid(theta, BLtheta_range[0], BLtheta_range[0] + side_rang_theta,
                                     BLtheta_range[1] - side_rang_theta, BLtheta_range[1])
            result_set["BLtheta"] = BLtheta
        if Ztheta_range[0]<theta and theta<Ztheta_range[1]:
            Ztheta = self.trapezoid(theta, Ztheta_range[0], Ztheta_range[0] + side_rang_theta,
                                    Ztheta_range[1] - side_rang_theta, Ztheta_range[1])
            result_set["Ztheta"] = Ztheta
        if BPtheta_range[0] < theta and theta < BPtheta_range[1]:
            BPtheta = self.trapezoid(theta, BPtheta_range[0], BPtheta_range[0] + side_rang_theta,
                                     BPtheta_range[1] - side_rang_theta, BPtheta_range[1])
            result_set["BPtheta"] = BPtheta
        if DPtheta_range[0] < theta and theta < DPtheta_range[1]:
            DPtheta = self.trapezoid(theta, DPtheta_range[0], DPtheta_range[0] + side_rang_theta, DPtheta_range[1],
                                     DPtheta_range[1])
            result_set["DPtheta"] = DPtheta

        """3. Okrślenie rozmywania wartości prędkości liniowej wózka"""
        # podzieliłem prędkość wózka na 5 przedziałów trapezowych
        side_rang_dx = 1
        V_range = [-100, -10, 0, 10, 100]
        WLX_range = [V_range[0], V_range[1] + side_rang_dx / 2]
        MLX_range = [V_range[1] - side_rang_dx / 2, V_range[2]]
        ZDX_range = [-side_rang_dx, side_rang_dx]
        MPX_range = [V_range[2], V_range[3] + side_rang_dx / 2]
        WPX_range = [V_range[3] - side_rang_dx / 2, V_range[4]]

        if WLX_range[0] < dx and dx < WLX_range[1]:
            WLX = self.trapezoid(dx, WLX_range[0], WLX_range[0], WLX_range[1] - side_rang_dx, WLX_range[1])
            result_set["WLX"] = WLX
        if MLX_range[0] < dx and dx < MLX_range[1]:
            MLX = self.trapezoid(dx, MLX_range[0], MLX_range[0] + side_rang_dx, MLX_range[1] - side_rang_dx,
                                 MLX_range[1])
            result_set["MLX"] = MLX
        if ZDX_range[0]<dx and dx<ZDX_range[1]:
            ZDX = self.trapezoid(dx, ZDX_range[0], ZDX_range[0] + side_rang_dx, ZDX_range[1] - side_rang_dx,
                                 ZDX_range[1])
            result_set["ZDX"] = ZDX
        if MPX_range[0] < dx and dx < MPX_range[1]:
            MPX = self.trapezoid(dx, MPX_range[0], MPX_range[0] + side_rang_dx, MPX_range[1] - side_rang_dx,
                                 MPX_range[1])
            result_set["MPX"] = MPX
        if WPX_range[0] < dx and dx < WPX_range[1]:
            WPX = self.trapezoid(dx, WPX_range[0], WPX_range[0] + side_rang_dx, WPX_range[1], WPX_range[1])
            result_set["WPX"] = WPX

        """4. Okrślenie rozmywania wartości prędkości kątowej"""
        # podzieliłem prędkość kątową wachdła na 5 przedziałów trapezowych
        side_rang_dtheta = sin(pi / 64)
        Vtheta_range = [-5, -sin(pi / 36), 0, sin(pi / 36), 5]
        WPtheta_range = [Vtheta_range[0], Vtheta_range[1] + side_rang_dtheta / 2]
        MPtheta_range = [Vtheta_range[1] - side_rang_dtheta / 2, Vtheta_range[2]]
        ZDtheta_range = [-side_rang_dtheta, side_rang_dtheta]
        MLtheta_range = [Vtheta_range[2], Vtheta_range[3] + side_rang_dtheta / 2]
        WLtheta_range = [Vtheta_range[3] - side_rang_dtheta / 2, Vtheta_range[4]]

        if WLtheta_range[0] < dtheta and dtheta < WLtheta_range[1]:
            WLtheta = self.trapezoid(dtheta, WLtheta_range[0], WLtheta_range[0], WLtheta_range[1] - side_rang_dtheta,
                                     WLtheta_range[1])
            result_set["WLtheta"] = WLtheta
        if MLtheta_range[0] < dtheta and dtheta < MLtheta_range[1]:
            MLtheta = self.trapezoid(dtheta, MLtheta_range[0], MLtheta_range[0] + side_rang_dtheta,
                                     MLtheta_range[1] - side_rang_dtheta, MLtheta_range[1])
            result_set["MLtheta"] = MLtheta
        if ZDtheta_range[0]<dtheta and dtheta<ZDtheta_range[1]:
            ZDtheta = self.trapezoid(dtheta, ZDtheta_range[0], ZDtheta_range[0] + side_rang_dtheta,
                                     ZDtheta_range[1] - side_rang_dtheta, ZDtheta_range[1])
            result_set["ZDtheta"] = ZDtheta
        if MPtheta_range[0] < dtheta and dtheta < MPtheta_range[1]:
            MPtheta = self.trapezoid(dtheta, MPtheta_range[0], MPtheta_range[0] + side_rang_dtheta,
                                     MPtheta_range[1] - side_rang_dtheta, MPtheta_range[1])
            result_set["MPtheta"] = MPtheta
        if WPtheta_range[0] < dtheta and dtheta < WPtheta_range[1]:
            WPtheta = self.trapezoid(dtheta, WPtheta_range[0], WPtheta_range[0] + side_rang_dtheta, WPtheta_range[1],
                                     WPtheta_range[1])
            result_set["WPtheta"] = WPtheta

        return result_set

    def fuzzy_NOT(self, arg):
        return 1 - arg

    def fuzzy_AND(self, *args):
        return min(list(args))

    def fuzzy_OR(self, *args):
        return max(list(args))

    def fuzzy_rules(self, result_set):
        rs = result_set
        force_dict = {"FBDL":0, "FDL": 0, "FML": 0, "FMP": 0, "FDP": 0,"FBDP":0}

        # FML
        force_dict["FML"] = self.fuzzy_OR(rs.get("BLtheta"),
                                          self.fuzzy_AND(rs.get("MLtheta"),rs.get("BPtheta"),self.fuzzy_NOT(rs.get("ZX"))),
                                          self.fuzzy_AND(rs.get("Ztheta"),rs.get("BLX"),rs.get("ZDtheta"),self.fuzzy_NOT(rs.get("ZX"))),

                                          self.fuzzy_AND(result_set.get("BPX"),result_set.get("BLtheta"),self.fuzzy_NOT(rs.get("ZX")))
                                          )
        # FMP
        force_dict["FMP"] = self.fuzzy_OR(rs.get("BPtheta"),
                                          self.fuzzy_AND(rs.get("MPtheta"),rs.get("BLtheta"),self.fuzzy_NOT(rs.get("ZX"))),
                                          self.fuzzy_AND(rs.get("Ztheta"),rs.get("BPX"),rs.get("ZDtheta"),self.fuzzy_NOT(rs.get("ZX"))),

                                          self.fuzzy_AND(result_set.get("BLX"),result_set.get("BPtheta"),self.fuzzy_NOT(rs.get("ZX")))
                                          )
        # FBDL
        force_dict["FBDL"] = self.fuzzy_OR(
                                        self.fuzzy_AND(self.fuzzy_NOT(rs.get("MPtheta")), rs.get("ZDX"), rs.get("Ztheta") )
                                        )
        # FBDP
        force_dict["FBDP"] = self.fuzzy_OR(
                                        self.fuzzy_AND(self.fuzzy_NOT(rs.get("MLtheta")), rs.get("ZDX"), rs.get("Ztheta") )
                                        )
        #FDL
        force_dict["FDL"]=self.fuzzy_OR(self.fuzzy_AND(result_set.get("DPX"), result_set.get("BLtheta")),
                                        self.fuzzy_AND(rs.get("MLX"),rs.get("ZDtheta"),rs.get("Ztheta"),rs.get("BPX"))
                                        )
        #FDP
        force_dict["FDP"] =self.fuzzy_OR(self.fuzzy_AND(result_set.get("DLX"),result_set.get("BPtheta")),
                                         self.fuzzy_AND(rs.get("MPX"),rs.get("ZDtheta"),rs.get("Ztheta"),rs.get("BLX"))
                                         )
        #print(result_set)
        #print(force_dict)
        return force_dict

    def defuzyfication(self, force):
        """Okrślenie wielkości siły"""
        FBDL = -200
        FDL = -50
        FML = -100
        FMP = 100
        FDP = 50
        FBDP = 200
        F = (force.get("FBDL")*FBDL+force.get("FDL")*FDL+force.get("FML")*FML+force.get("FMP")*FMP+
             force.get("FDP")*FDP+force.get("FBDP")*FBDP)/(sum(force.values()))
        #print(f"Siła: {F}")
        return F


if __name__ == '__main__':
        if len(argv)>1:
            ip = InvertedPendulum(f_name=argv[1])
        else:
            ip = InvertedPendulum(x0=90, dx0=0, theta0=0, dtheta0=0.1, ih=800, iw=1000, h_min=-80, h_max=80)
        ip.run(sandbox=False)
