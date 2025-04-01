import numpy as np
import qutip as q

# -------------------------------------------------

def calc_miri_list(env, t_end, t_steps, pauli=True, old_register_states=None):
    HGate = 1j * env.calc_U_rot(np.pi, 0, theta=np.pi / 4)
    HGate_phase = 1j * env.calc_U_rot(np.pi, 0, theta=np.pi / 4) # np.pi/2

    # time list
    t_list = np.linspace(0, t_end, t_steps)
    env.gate_props_list = [("free_evo", dict(t=t_end))]

    # initial states
    xm, xp = q.sigmax().eigenstates()[1]
    xp, xm = xp * xp.dag(), xm * xm.dag()
    ym, yp = q.sigmay().eigenstates()[1]
    yp, ym = yp * yp.dag(), ym * ym.dag()
    zp, zm = q.sigmaz().eigenstates()[1]
    zm, zp = zp * zp.dag(), zm * zm.dag()
    if old_register_states is None:
        old_register_states = [HGate * dm * HGate for dm in [xp, xm, yp, ym, zp, zm]]
    else: 
        old_register_states = [HGate * dm * HGate for dm in old_register_states]

    states = env.calc_states(t_list=t_list, old_register_states=old_register_states)
    states = np.array([[HGate_phase * dm * HGate_phase for dm in row] for row in states])

    if pauli:
        sigmax = np.array(
            [[q.expect(dm, q.sigmax()).real for dm in row] for row in states]
        )
        sigmay = np.array(
            [[q.expect(dm, q.sigmay()).real for dm in row] for row in states]
        )
        sigmaz = np.array(
            [[q.expect(dm, q.sigmaz()).real for dm in row] for row in states]
        )

        miri_list = np.array([sigmax, sigmay, sigmaz])

    else:
        pop0 = np.array(
            [[q.expect(dm, q.fock_dm(2, 0)).real for dm in row] for row in states]
        )
        pop1 = np.array(
            [[q.expect(dm, q.fock_dm(2, 1)).real for dm in row] for row in states]
        )

        miri_list = np.array([pop0, pop1])
    
    return np.transpose(miri_list, (1, 0, 2))


def calc_miri_list2(env, t_end, t_steps, pauli=True):

    # initial states
    xp, xm = q.sigmax().eigenstates()[1]
    xp, xm = xp * xp.dag(), xm * xm.dag()
    yp, ym = q.sigmay().eigenstates()[1]
    yp, ym = yp * yp.dag(), ym * ym.dag()
    zp, zm = q.sigmaz().eigenstates()[1]
    zp, zm = zp * zp.dag(), zm * zm.dag()
    old_register_states = [xp, xm, yp, ym, zp, zm]

    # time list
    t_list = np.linspace(0, t_end, t_steps)

    num_obs = 3 if pauli else 2
    miri_list = np.zeros((t_steps, num_obs, 6))

    for i, t in enumerate(t_list):
        env.gate_props_list = [
            ("inst_rot", dict(alpha=np.pi, phi=0, theta=np.pi / 4)),
            ("free_evo", dict(t=t)),
            ("inst_rot", dict(alpha=np.pi, phi=0, theta=np.pi / 4)),
        ]

        states = env.calc_states(
            t_list="final", old_register_states=old_register_states
        )[:, 0]
        # states = np.array([HGate * dm * HGate for dm in states])

        if pauli:
            sigmax = np.array([q.expect(dm, q.sigmax()) for dm in states])
            sigmay = np.array([q.expect(dm, q.sigmay()) for dm in states])
            sigmaz = np.array([q.expect(dm, q.sigmaz()) for dm in states])

            miri_list[i] = np.array([sigmax, sigmay, sigmaz])
        else:
            pop0 = np.array([q.expect(dm, q.fock_dm(2, 0)) for dm in states])
            pop1 = np.array([q.expect(dm, q.fock_dm(2, 1)) for dm in states])

            miri_list[i] = np.array([pop0, pop1])
    miri_list = np.transpose(miri_list, (1, 2, 0))
    return miri_list
