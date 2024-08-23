from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
import numpy as np
import matplotlib.pyplot as plt

class SimpleTurbine(WindTurbine):
    def __init__(self, method='linear'):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        # WindTurbine.__init__(self, name='wt', diameter=1, hub_height=1,
        #                         powerCtFunction=PowerCtTabular(ws=[0, 3, 12, 25, 30],
        #                                                         power=[0, 0, 2000, 2000, 0],
        #                                                         power_unit="kw",
        #                                                         ct=[0, 8/9, 8/9, 0.3, 0],
        #                                                         method=method))
        
        WindTurbine.__init__(self, name='wt', diameter=1, hub_height=1,
                                powerCtFunction=PowerCtTabular(ws=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                                                                power=[0.0, 66.6, 154.0, 282.0, 460.0, 696.0, 996.0, 1341.0, 1661.0, 1866.0, 1958.0, 1988.0, 1997.0, 1999.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
                                                                power_unit="kw",
                                                                ct=[0.0, 0.818, 0.806, 0.804, 0.805, 0.806, 0.807, 0.793, 0.739, 0.709, 0.409, 0.314, 0.249, 0.202, 0.167, 0.14, 0.119, 0.102, 0.088, 0.077, 0.067, 0.06, 0.053],
                                                                method=method))
if __name__ == "__main__":
    from py_wake.examples.data.hornsrev1 import V80
    wt = SimpleTurbine()
    # wt = V80()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.linspace(0, 25, 100)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()  # Create a second y-axis

    ax1.plot(ws, wt.power(ws) * 1e-3, color='b', label='Power')
    ax1.set_xlabel('Wind speed [m/s]')
    ax1.set_ylabel('Power output [kW]')
    # ax1.tick_params('y', colors='b')

    ax2.plot(ws, wt.ct(ws), color="r", label='CT')
    ax2.set_ylabel('Thrust coefficient [-]')
    # ax2.tick_params('y', colors="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig.tight_layout()
    plt.show()