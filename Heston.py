from scipy.stats import norm
import numpy as np
from HestonMC import MC
from time import time

def C_Heston(params, spot, strike, tau, r):
    sigma_0, k, theta, nu, rho = params
    
    sigma_0_2 = sigma_0 ** 2
    x = np.log(spot)
    x_star = np.log(strike) - r * tau
    D_t = np.exp(-k * tau)
    v_t = np.sqrt(theta + (sigma_0_2 - theta) / (k * tau) * (1 - D_t))
    d_plus = (x - x_star) / (v_t * np.sqrt(tau)) + 1 / 2 * v_t * np.sqrt(tau)
    d_minus = d_plus - v_t * np.sqrt(tau)
    BS = np.exp(x) * norm.cdf(d_plus) - strike * np.exp(-r * tau) * norm.cdf(d_minus)

    R_t = nu ** 2 / (8 * k ** 2) * (
            theta * tau + (sigma_0_2 - theta) / k * (1 - D_t) - 2 * theta / k * (1 - D_t) - 2 * (
            sigma_0_2 - theta) * tau * D_t + theta / (2 * k) * (1 - np.exp(-2 * k * tau)) + (
                    sigma_0_2 - theta) / k * (D_t - np.exp(-2 * k * tau)))
    K = np.exp(x) / (v_t * np.sqrt(2 * np.pi * tau)) * np.exp(-1 / 2 * d_plus ** 2) * (
            (d_plus ** 2 - v_t * d_plus * np.sqrt(tau) - 1) / (tau * v_t ** 2))

    U_t = rho * nu / (2 * k ** 2) * (
            theta * k * tau - 2 * theta + sigma_0_2 + D_t * (2 * theta - sigma_0_2) - k * tau * D_t * (
            sigma_0_2 - theta))
    H = np.exp(x) / (v_t * np.sqrt(2 * np.pi * tau)) * np.exp(-1 / 2 * d_plus ** 2) * (
            1 - d_plus / (v_t * np.sqrt(tau)))

    return BS + H * U_t + K * R_t


def fHes(p, S, K, T, r):
    def HesIntMN(u):
        csqr = nu ** 2
        PQ_M = P + Q * u
        PQ_N = P - Q * u

        imPQ_M = 1j * PQ_M
        imPQ_N = 1j * PQ_N
        _imPQ_M = 1j * (PQ_M - 1j)
        _imPQ_N = 1j * (PQ_N - 1j)

        h_M = pow(K, -imPQ_M) / imPQ_M
        h_N = pow(K, -imPQ_N) / imPQ_N

        x0 = np.log(S) + r * T
        # kes = a-i*c*rho*u1;
        tmp = nu * rho
        kes_M1 = k - tmp * _imPQ_M
        kes_N1 = k - tmp * _imPQ_N
        kes_M2 = kes_M1 + tmp
        kes_N2 = kes_N1 + tmp

        # m = i * u1 + pow(u1, 2)
        # ONES AND ZEROES
        m_M1 = imPQ_M + 1 + pow(PQ_M - 1j, 2)  # m_M1 = (PQ_M - 1j) * 1j + pow(PQ_M - 1j, 2);
        m_N1 = imPQ_N + 1 + pow(PQ_N - 1j, 2)  # m_N1 = (PQ_N - 1j) * 1j + pow(PQ_N - 1j, 2);
        m_M2 = imPQ_M + pow(PQ_M - 0 * 1j, 2)
        m_N2 = imPQ_N + pow(PQ_N - 0 * 1j, 2)

        # d = sqrt(pow(kes, 2) + m * pow(c, 2))
        d_M1 = np.sqrt(pow(kes_M1, 2) + m_M1 * csqr)
        d_N1 = np.sqrt(pow(kes_N1, 2) + m_N1 * csqr)
        d_M2 = np.sqrt(pow(kes_M2, 2) + m_M2 * csqr)
        d_N2 = np.sqrt(pow(kes_N2, 2) + m_N2 * csqr)

        # g = exp(-a * b * rho * T * u1 * i / c)
        tmp1 = -k * theta * rho * T / nu
        tmp = np.exp(tmp1)
        g_M2 = np.exp(tmp1 * imPQ_M)
        g_N2 = np.exp(tmp1 * imPQ_N)
        g_M1 = g_M2 * tmp
        g_N1 = g_N2 * tmp

        # alp, calp, salp
        tmp = 0.5 * T
        alpha = d_M1 * tmp
        calp_M1 = np.cosh(alpha)
        salp_M1 = np.sinh(alpha)

        alpha = d_N1 * tmp
        calp_N1 = np.cosh(alpha)
        salp_N1 = np.sinh(alpha)

        alpha = d_M2 * tmp
        calp_M2 = np.cosh(alpha)
        salp_M2 = np.sinh(alpha)

        alpha = d_N2 * tmp
        calp_N2 = np.cosh(alpha)
        salp_N2 = np.sinh(alpha)

        # A2 = d*calp + kes*salp
        A2_M1 = d_M1 * calp_M1 + kes_M1 * salp_M1
        A2_N1 = d_N1 * calp_N1 + kes_N1 * salp_N1
        A2_M2 = d_M2 * calp_M2 + kes_M2 * salp_M2
        A2_N2 = d_N2 * calp_N2 + kes_N2 * salp_N2

        # A1 = m*salp
        A1_M1 = m_M1 * salp_M1
        A1_N1 = m_N1 * salp_N1
        A1_M2 = m_M2 * salp_M2
        A1_N2 = m_N2 * salp_N2

        # A = A1/A2
        A_M1 = A1_M1 / A2_M1
        A_N1 = A1_N1 / A2_N1
        A_M2 = A1_M2 / A2_M2
        A_N2 = A1_N2 / A2_N2

        # characteristic function: y1 = exp(i * x0 * u1) * exp(-v0 * A) * g * exp(2 * a * b / pow(c, 2) * D)
        tmp = 2 * k * theta / csqr
        halft = 0.5 * T
        D_M1 = np.log(d_M1) + (k - d_M1) * halft - np.log(
            (d_M1 + kes_M1) * 0.5 + (d_M1 - kes_M1) * 0.5 * np.exp(-d_M1 * T))
        D_M2 = np.log(d_M2) + (k - d_M2) * halft - np.log(
            (d_M2 + kes_M2) * 0.5 + (d_M1 - kes_M2) * 0.5 * np.exp(-d_M2 * T))
        D_N1 = np.log(d_N1) + (k - d_N1) * halft - np.log(
            (d_N1 + kes_N1) * 0.5 + (d_N1 - kes_N1) * 0.5 * np.exp(-d_N1 * T))
        D_N2 = np.log(d_N2) + (k - d_N2) * halft - np.log(
            (d_N2 + kes_N2) * 0.5 + (d_N2 - kes_N2) * 0.5 * np.exp(-d_N2 * T))

        M1 = np.real(h_M * np.exp(x0 * _imPQ_M - sigma_0 * A_M1 + tmp * D_M1) * g_M1)
        N1 = np.real(h_N * np.exp(x0 * _imPQ_N - sigma_0 * A_N1 + tmp * D_N1) * g_N1)
        M2 = np.real(h_M * np.exp(x0 * imPQ_M - sigma_0 * A_M2 + tmp * D_M2) * g_M2)
        N2 = np.real(h_N * np.exp(x0 * imPQ_N - sigma_0 * A_N2 + tmp * D_N2) * g_N2)

        return M1, N1, M2, N2

    lb = 0.0
    ub = 200
    Q = 0.5 * (ub - lb)
    P = 0.5 * (ub + lb)

    u = [0.0243502926634244325089558,0.0729931217877990394495429,0.1214628192961205544703765,0.1696444204239928180373136,0.2174236437400070841496487,0.2646871622087674163739642,0.3113228719902109561575127,0.3572201583376681159504426,0.4022701579639916036957668,0.4463660172534640879849477,0.4894031457070529574785263,0.5312794640198945456580139,0.5718956462026340342838781,0.6111553551723932502488530,0.6489654712546573398577612,0.6852363130542332425635584,0.7198818501716108268489402,0.7528199072605318966118638,0.7839723589433414076102205,0.8132653151227975597419233,0.8406292962525803627516915,0.8659993981540928197607834,0.8893154459951141058534040,0.9105221370785028057563807,0.9295691721319395758214902,0.9464113748584028160624815,0.9610087996520537189186141,0.9733268277899109637418535,0.9833362538846259569312993,0.9910133714767443207393824,0.9963401167719552793469245,0.9993050417357721394569056]
    w = [0.0486909570091397203833654,0.0485754674415034269347991,0.0483447622348029571697695,0.0479993885964583077281262,0.0475401657148303086622822,0.0469681828162100173253263,0.0462847965813144172959532,0.0454916279274181444797710,0.0445905581637565630601347,0.0435837245293234533768279,0.0424735151236535890073398,0.0412625632426235286101563,0.0399537411327203413866569,0.0385501531786156291289625,0.0370551285402400460404151,0.0354722132568823838106931,0.0338051618371416093915655,0.0320579283548515535854675,0.0302346570724024788679741,0.0283396726142594832275113,0.0263774697150546586716918,0.0243527025687108733381776,0.0222701738083832541592983,0.0201348231535302093723403,0.0179517157756973430850453,0.0157260304760247193219660,0.0134630478967186425980608,0.0111681394601311288185905,0.0088467598263639477230309,0.0065044579689783628561174,0.0041470332605624676352875,0.0017832807216964329472961]
    NumGrids = len(u)
    # k, theta, nu, rho, sigma_0 = p
    sigma_0, k, theta, nu, rho = p
    disc = np.exp(-r*T)
    tmp = 0.5 * (S - K * disc)
    disc = disc/np.pi
    Y1, Y2 = 0.0, 0.0

    for j in range(NumGrids):
        MN = HesIntMN(u[j])
        M1, N1, M2, N2 = MN
        Y1 += w[j]*(M1+N1)
        Y2 += w[j] * (M2 + N2)

    Qv1 = Q * Y1
    Qv2 = Q * Y2
    pv = tmp + disc * (Qv1 - K * Qv2)

    return pv




if __name__ == '__main__':
    np.random.seed(42)

    # sigma_0 = 0.1197    # Initial variance is square of volatility
    # k = 1.98937         # Mean reversion rate
    # theta = 0.3 ** 2    # Long-run variance
    # nu = 0.33147        # Volatility of volatility
    # rho = -0.45648749   # Correlation
    # params = [sigma_0, k, theta, nu, rho]

    # params = [ 0.79786259,  5.,          0.4579982,   2.14008933, -0.1862887 ]
    # S, strike, tau, r = 4299.68, 5750, 0.0201548679921377, 0.0235
    #
    # start = time()
    # print(f'C_Heston = {C_Heston(params, S, strike, tau, r)} time: {time() - start}')
    # print(f'MC = {MC(params, S, strike, tau, r)}')


    lb = 0.0
    ub = 200
    Q = 0.5 * (ub - lb)
    P = 0.5 * (ub + lb)
    params = [ 0.79786259,  5.,          0.4579982,   2.14008933, -0.1862887 ]
    S, strike, tau, r = 4299.68, 5750, 0.0201548679921377, 0.0235
    start = time()
    print(f'MC = {MC(params, S, strike, tau, r, nsim=100000, N=1000)}  time: {time() - start}')
    start = time()
    print(f'C_Heston = {C_Heston(params, S, strike, tau, r)} time: {time() - start}')
    start = time()
    print(f'fHes = {fHes(params, S, strike, tau, r)} time: {time() - start}')
    pass
