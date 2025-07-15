from value import Value

# using cross entropy loss for cost function
def softmax(logits):
    exps = [x.exp() for x in logits]
    sum_exps = sum(exps, Value(0))
    probs = [e / sum_exps for e in exps]
    return probs

def cross_entropy_loss(logits, true_label):
    probs = softmax(logits)
    loss = Value(0)
    for p, t in zip(probs, true_label):
        if t == 1:
            loss -= p.log()
    return loss

def optimizer(model, base_lr, e, decay=0.96):
    lr = base_lr * (decay ** e)
    for p in model.parameters():
         p.data += -lr * p.grad