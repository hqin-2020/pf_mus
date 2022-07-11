import numpy as np
import scipy as sp
import seaborn as sns
sns.set()

def simulate(θ_true, T):
    
    Azo = θ_true[0]; Azz = θ_true[1]; Bz = θ_true[2]
    Aso = θ_true[3]; Ass = θ_true[4]; Bs = θ_true[5]
    
    Z01 = 0
    Z02 = Azo[1,0]/(1-Azz[1,1])
    S0 = sp.linalg.solve((np.eye(3) - Ass), Aso)

    Z = np.zeros((2,T+1)) 
    S = np.zeros((3,T+1)) 
    Z[:,[0]] = np.array([[Z01],[Z02]])
    S[:,[0]] = S0

    np.random.seed(0)
    Wz = np.random.multivariate_normal(np.zeros(2), np.eye(2), T+1).T
    np.random.seed(1)
    Ws = np.random.multivariate_normal(np.zeros(3), np.eye(3), T+1).T

    for t in range(T):
        Z[:,[t+1]] = Azo + Azz @ Z[:,[t]] + Bz @ Wz[:,[t+1]]
        S[:,[t+1]] = Aso + Ass @ S[:,[t]] + Bs @ Ws[:,[t+1]]

    D = np.ones((3,1)) @ Z[[0],:] + S
    
    return D

def decompose_θ(θ):
    
    λ = θ[1][1,1]; η = θ[0][1,0]
    b11 = θ[2][0,0]; b22 = θ[2][1,1]
    Aso2 = θ[3][1,0]; Aso3 = θ[3][2,0]
    As11 = θ[4][0,0]; As12 = θ[4][0,1]; As13 = θ[4][0,2]
    As21 = θ[4][1,0]; As22 = θ[4][1,1]; As23 = θ[4][1,2]
    As31 = θ[4][2,0]; As32 = θ[4][2,1]; As33 = θ[4][2,2]
    Bs11 = θ[5][0,0];  Bs21 = θ[5][1,0];  Bs22 = θ[5][1,1]; Bs31 = θ[5][2,0]; Bs32 = θ[5][2,1];  Bs33 = θ[5][2,2]
    
    (P, L, U) = sp.linalg.lu(θ[5]@θ[5].T)
    D = np.diag(np.diag(U))   # D is just the diagonal of U
    U /= np.diag(U)[:, None]  # Normalize rows of U
    J = L
    Δ = D
    J_inv = sp.linalg.inv(J)
    j21 = J_inv[1,0]; j31 = J_inv[2,0]; j32 = J_inv[2,1]

    return λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33, j21, j31, j32
   
def update_θ(H):

    stability_max_iter = 200_000
    
    β_z1, σ2_z1 = draw_para(H[0][0], H[0][1], H[0][2], H[0][3])
    b11 = np.sqrt(σ2_z1)
    
    λ_stable = False
    λ_iter = 0
    while λ_stable == False:
        β_z2, σ2_z2 = draw_para(H[1][0], H[1][1], H[1][2], H[1][3])
        η = β_z2[0,0] 
        λ = β_z2[1,0] 
        b22 = np.sqrt(σ2_z2)
        λ_iter += 1
        if abs(λ) < 0.99:
            λ_stable = True
        elif λ_iter > stability_max_iter:
            print('λ Unstable')
        else:
            λ_stable = False

    Azo = np.array([[0],[η]])
    Azz = np.array([[1, 1],[0, λ]])
    Bz = np.array([[b11, 0],[0, b22]])

    Ass_stable = False
    Ass_iter = 0
    while Ass_stable == False:
        β_s1, σ2_s1 = draw_para(H[2][0], H[2][1], H[2][2], H[2][3])
        β_s2, σ2_s2 = draw_para(H[3][0], H[3][1], H[3][2], H[3][3])
        β_s3, σ2_s3 = draw_para(H[4][0], H[4][1], H[4][2], H[4][3])

        J_inv = np.array([[1.0,                0.0,        0.0],\
                          [-β_s2[1,0],         1.0,        0.0],\
                          [-β_s3[1,0],  -β_s3[2,0],        1.0]])
        Aso = sp.linalg.solve(J_inv, np.array([[0], β_s2[0], β_s3[0]]))
        Ass = sp.linalg.solve(J_inv, np.array([[β_s1[0,0], β_s1[1,0], β_s1[2,0]],\
                                               [β_s2[2,0], β_s2[3,0], β_s2[4,0]],\
                                               [β_s3[3,0], β_s3[4,0], β_s3[5,0]]]))
        Ass_iter += 1
        if np.max(abs(np.linalg.eigvals(Ass)))<0.99:
            Ass_stable = True
        elif Ass_iter > stability_max_iter:
            print('Ass Unstable')
        else:
            Ass_stable = False
    
    Bs = sp.linalg.solve(J_inv, np.diag([σ2_s1,σ2_s2,σ2_s3])**0.5)
    θ = [Azo, Azz, Bz, Aso, Ass, Bs, J_inv, λ_iter, Ass_iter]
    
    return θ

def init_X(θ, D_0):
    
    λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33, j21, j31, j32 = decompose_θ(θ)
    ones = np.ones([3,1])
    Ass = np.array([[As11, As12, As13],\
                    [As21, As22, As23],\
                    [As31, As32, As33]])
    Aso = np.array([[0.0],\
                    [Aso2],\
                    [Aso3]])
    Bs =  np.array([[Bs11, 0,    0],\
                    [Bs21, Bs22, 0],\
                    [Bs31, Bs32, Bs33]])
    
    μs = sp.linalg.solve(np.eye(3) - Ass, Aso) 
    Σs = sp.linalg.solve_discrete_lyapunov(Ass, Bs@Bs.T)
    
    β = sp.linalg.solve(np.hstack([Σs@np.array([[1,1],[0,-1],[-1,0]]), ones]).T, np.array([[0,0,1]]).T)                                     
    γ1 = np.array([[1],[0],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ2 = np.array([[0],[1],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ3 = np.array([[0],[0],[1]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    Γ = np.hstack([γ1, γ2, γ3])
    
    Z01 = β.T@(D_0 - μs)
    Σz01 = 0.0
    Z02 = η/(1-λ)
    Σz02 = b22**2/(1-λ**2)
    S0 = Γ.T@(D_0 - μs) + μs
    Σs0 = (1/(ones.T@sp.linalg.inv(Σs)@ones))[0][0]
    
    μ0 = np.array([[Z01[0][0]],\
                   [Z02],\
                   [S0[0][0]],\
                   [S0[1][0]],\
                   [S0[2][0]]])
    Σ0 = np.array([[Σz01,0.0,    0.0,   0.0,   0.0],\
                   [0.0,   Σz02, 0.0,   0.0,   0.0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0]]) 
    return μ0, Σ0


def init(Input):
    D0, seed = Input
    np.random.seed(seed)
    H0 = [[np.array([[1.0], [1.0]]), np.eye(2), 1, 1],\
          [np.array([[0.0], [0.0]]), np.eye(2), 1, 1],\
          [np.array([[0.0], [0.0], [0.0]]), np.eye(3), 1, 1],\
          [np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]), np.eye(5), 1, 1],\
          [np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]), np.eye(6), 1, 1]]
    
    Σ0_postive = False
    while Σ0_postive == False:
        θ0 = update_θ(H0)
        μ0, Σ0 = init_X(θ0, D0)
        if np.all(np.linalg.eigvals(Σ0)>=0) == True:
            Σ0_postive = True
        else: 
            Σ0_postive = False
    
    X0 = sp.stats.multivariate_normal.rvs(μ0.flatten(), Σ0).reshape(-1,1)
    ν0 = 1

    return [θ0, X0, H0, ν0]

def bayes_para_update(bt, Λt, ct, dt, Rt_next, Zt_next):
    
    Λt_next = Λt + Rt_next@Rt_next.T
    bt_next = np.linalg.solve(Λt_next, Λt@bt + Rt_next*Zt_next)
    ct_next = ct + 1.0
    dt_next = Zt_next**2 - bt_next.T@Λt_next@bt_next + bt.T@Λt@bt + dt
    
    return bt_next, Λt_next, ct_next, dt_next

def draw_para(b, Λ, c, d):
    
    ζ = sp.stats.gamma.rvs(c/2+1, loc = 0, scale = 1/(d/2))
    σ2 = 1/ζ
    β = np.random.multivariate_normal(b.flatten(), sp.linalg.inv(ζ*Λ)).reshape(-1,1)
    return β, σ2

def recursive(Input):
    
    Dt_next, Xt, Ht, seed = Input
    np.random.seed(seed)
    stability_max_iter = 200_000
    
    β_z1, σ2_z1 = draw_para(Ht[0][0], Ht[0][1], Ht[0][2], Ht[0][3])
    b11 = np.sqrt(σ2_z1)
    
    λ_stable = False
    λ_iter = 0
    while λ_stable == False:
        β_z2, σ2_z2 = draw_para(Ht[1][0], Ht[1][1], Ht[1][2], Ht[1][3])
        η = β_z2[0,0] 
        λ = β_z2[1,0] 
        b22 = np.sqrt(σ2_z2)
        λ_iter += 1
        if abs(λ) < 0.99:
            λ_stable = True
        elif λ_iter > stability_max_iter:
            print('λ Unstable')
        else:
            λ_stable = False

    Azo = np.array([[0],[η]])
    Azz = np.array([[1, 1],[0, λ]])
    Bz = np.array([[b11, 0],[0, b22]])
    
    Ass_stable = False
    Ass_iter = 0
    
    while Ass_stable == False:
        β_s1, σ2_s1 = draw_para(Ht[2][0], Ht[2][1], Ht[2][2], Ht[2][3])
        β_s2, σ2_s2 = draw_para(Ht[3][0], Ht[3][1], Ht[3][2], Ht[3][3])
        β_s3, σ2_s3 = draw_para(Ht[4][0], Ht[4][1], Ht[4][2], Ht[4][3])

        J_inv = np.array([[1.0,                0.0,        0.0],\
                          [-β_s2[1,0],         1.0,        0.0],\
                          [-β_s3[1,0], -β_s3[2,0],         1.0]])
        Aso = sp.linalg.solve(J_inv, np.array([[0], β_s2[0], β_s3[0]]))
        Ass = sp.linalg.solve(J_inv, np.array([[β_s1[0,0], β_s1[1,0], β_s1[2,0]],\
                                               [β_s2[2,0], β_s2[3,0], β_s2[4,0]],\
                                               [β_s3[3,0], β_s3[4,0], β_s3[5,0]]]))
        Ass_iter += 1
        if np.max(abs(np.linalg.eigvals(Ass)))<0.99:
            Ass_stable = True
        elif Ass_iter > stability_max_iter:
            print('Ass Unstable')
        else:
            Ass_stable = False
    
    Bs = sp.linalg.solve(J_inv, np.diag([σ2_s1,σ2_s2,σ2_s3])**0.5)
    θt_next = [Azo, Azz, Bz, Aso, Ass, Bs, J_inv, λ_iter, Ass_iter]
    
    Bz1 = Bz[[0],:]
    Zt = Xt[0:2,:]; Zt1 = Zt[0,0]; Zt2 = Zt[1,0]
    St = Xt[2:5,:]
    ones = np.ones([3,1])

    Φ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, ones@Bz1@Bz.T)
    Γ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, Bs@Bs.T)
    
    mean = np.vstack([Azo + Azz@Zt + Φ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St),\
                      Aso + Ass@St + Γ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St)])
    cov = np.vstack([np.hstack([Bz@Bz.T, np.zeros([2,3])]),\
                     np.hstack([np.zeros([3,2]), Bs@Bs.T])]) -\
          np.vstack([Φ.T, Γ.T]) @ (ones@Bz1@Bz1.T@ones.T+Bs@Bs.T)@np.hstack([Φ, Γ])

    Xt_next = sp.stats.multivariate_normal.rvs(mean.flatten(), cov).reshape(-1,1)
    
    St1 = Xt[2,0]; St2 = Xt[3,0]; St3 = Xt[4,0];   
    Zt_next_1 = Xt_next[0,0];  Zt_next_2 = Xt_next[1,0];  St_next_1 = Xt_next[2,0];  St_next_2 = Xt_next[3,0];  St_next_3 = Xt_next[4,0];  
    
    first_eq_Rt_next = np.array([[Zt1],[Zt2]])
    first_eq_Zt_next = Zt_next_1
    first_eq_bt_next = np.array([[1.0],[1.0]])
    first_eq_Λt_next = Ht[0][1] + first_eq_Rt_next@first_eq_Rt_next.T
    first_eq_ct_next = Ht[0][2] + 1.0
    first_eq_dt_next = (first_eq_Zt_next - first_eq_Rt_next.T@first_eq_bt_next)**2  + Ht[0][3]
    
    second_eq_bt_next, second_eq_Λt_next, second_eq_ct_next, second_eq_dt_next = \
    bayes_para_update(Ht[1][0], Ht[1][1], Ht[1][2], Ht[1][3], np.array([[1],[Zt2]]), Zt_next_2)
    
    third_eq_bt_next, third_eq_Λt_next, third_eq_ct_next, third_eq_dt_next = \
    bayes_para_update(Ht[2][0], Ht[2][1], Ht[2][2], Ht[2][3], np.array([[St1],[St2],[St3]]), St_next_1)
    
    fourth_eq_bt_next, fourth_eq_Λt_next, fourth_eq_ct_next, fourth_eq_dt_next = \
    bayes_para_update(Ht[3][0], Ht[3][1], Ht[3][2], Ht[3][3], np.array([[1],[St_next_1],[St1],[St2],[St3]]), St_next_2)
    
    fifth_eq_bt_next, fifth_eq_Λt_next, fifth_eq_ct_next, fifth_eq_dt_next = \
    bayes_para_update(Ht[4][0], Ht[4][1], Ht[4][2], Ht[4][3], np.array([[1],[St_next_1],[St_next_2],[St1],[St2],[St3]]), St_next_3)
    
    Ht_next = [[first_eq_bt_next, first_eq_Λt_next, first_eq_ct_next, first_eq_dt_next],\
               [second_eq_bt_next, second_eq_Λt_next, second_eq_ct_next, second_eq_dt_next],\
               [third_eq_bt_next, third_eq_Λt_next, third_eq_ct_next, third_eq_dt_next],\
               [fourth_eq_bt_next, fourth_eq_Λt_next, fourth_eq_ct_next, fourth_eq_dt_next],\
               [fifth_eq_bt_next, fifth_eq_Λt_next, fifth_eq_ct_next, fifth_eq_dt_next]]

    des_mean = ones*Zt1 +ones*Zt2 + Aso + Ass@St
    des_cov = ones@Bz1@Bz1.T@ones.T+Bs@Bs.T
    
    density = sp.stats.multivariate_normal.pdf(Dt_next.flatten(), des_mean.flatten(), des_cov)
    
    return [θt_next, Xt_next, Ht_next, density]