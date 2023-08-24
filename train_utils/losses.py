import torch
import torch.nn.functional as F

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def FDM_nonsep_u(rho, u, D=1):
    n_samples, nt, nx = rho.shape
    V = torch.zeros([n_samples, nt + 1, nx + 1], dtype=torch.float32)
    u = torch.zeros([n_samples, nt, nx], dtype=torch.float32)
    dx, dt = 1 / nx, 1 / nt
    for t in range(nt - 1, -1, -1):
        for i in range(nx):
            for j, s in enumerate((V[:, t + 1, i] - V[:, t + 1, i + 1]) / dx + 1 - rho[:, t, i]):
                u[j, t, i] = min(max(s, 0), 1)

            # Non-sep:
            V[:, t, i] = dt * (0.5 * u[:, t, i] ** 2 + rho[:, t, i] * u[:, t, i] - u[:, t, i]) + \
                          (1 - u[:, t, i]) * V[:, t + 1, i] + u[:, t, i] * V[:, t + 1, i + 1]
            # LWR:
            # V[:, t, i] = (delta_t * 0.5 * (1 - u[:, t, i] - rho[:, t, i])** 2).to("cuda:0") + \
            #              (1 - u[:, t, i]) * V[:, t + 1, i] + u[:, t, i] * V[:, t + 1, i + 1]

        V[:, t, -1] = V[:, t, 0]

    V_terminal = 0
    V[:, -1, :] = V_terminal

    u = u.detach()
    rhot = (rho[:, 1:, :] - rho[:, :-1, :]) / dt
    rhox = (rho[:, :, :] - torch.cat((rho[:, :, [-1]], rho[:, :, :-1]), -1)) / dx
    ux = (u[:, :, :] - torch.cat((u[:, :, [-1]], u[:, :, :-1]), -1)) / dx

    f = rhot + (rhox * u + rho * ux)[:, 1:, :]
    return f

def FDM_nonsep_V(V, rho, D=1):

    n_samples, nt, nx = rho.shape
    dx, dt = 1 / nx, 1 / nt

    Vt = (V[:, 1:, :] - V[:, :-1, :]) / dt
    Vx = (V[:, :, 1:] - V[:, :, :-1]) / dx

    u = 1 - rho - Vx
    f = Vt[:, :, 1:] + (u * Vx + 0.5 * u ** 2 - u + u * rho)[:, 1:, :]

    return f


def PINO_loss_rho(u, u0, c):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)

    index_t = torch.zeros(nx,).long()

    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_nonsep_u(u, c)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    return loss_u, loss_f

def PINO_loss_V(u, u0, c):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)

    index_t = torch.zeros(nx,).long()
    index_t = torch.full_like(index_t, nt - 1) # for V add boundary for the last line of T

    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_nonsep_V(u, c)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    return loss_u, loss_f