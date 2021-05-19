import matplotlib.pyplot as plt
from torch import save
from os.path import join


class Logger:
    def __init__(self,
                 save_dir,
                 total_epoch=0):

        self.loss_log_list = {
            "train": [],
            "valid": []
        }
        self.save_dir = save_dir
        self.total_batches_done = 0
        self.total_epoch = total_epoch
        self.current_epoch = 1

    def print_loss(self, phase="train"):
        log_fmt = "[{}] Epoch - {}/{} | Batches Done - {} | Loss - {:.5f}"
        print(log_fmt.format(phase,
                             self.current_epoch + 1,
                             self.total_epoch,
                             self.total_batches_done,
                             self.loss_log_list[phase][-1]))

    def plot_loss(self, phase="train", title="Loss"):
        plt.figure(figsize=(15, 10))
        plt.title(title)
        plt.plot(self.loss_log_list[phase], 'g-')
        plt.savefig(join(self.save_dir, f"{phase}_loss.jpg"), dpi=80, bbox_inches="tight")
        plt.close()

    def save_model(self,
                   model,
                   optimizer,
                   epoch,
                   loss: float,
                   best=False):

        save_dict = {
            "epoch": epoch,
            "batches_done": self.total_batches_done,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        save_path = join(self.save_dir, "checkpoint.pth")
        save(save_dict, save_path)
        if best:
            save_path = join(self.save_dir, "best.pth")
            save(save_dict, save_path)
            print("Best model updated")

        print("save model to {}\n".format(save_path))


if __name__ == '__main__':
    pass
