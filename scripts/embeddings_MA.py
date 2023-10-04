# tutorial from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import pathlib 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def plot_loss(losses, n_epochs, savepath):
    # define figure size 
    plt.figure(figsize=(12,6))

    # create plot of train and validation loss, defined as two subplots on top of each other ! (but beside the accuracy plot)
    plt.plot(np.arange(1, n_epochs+1), losses, label="train_loss") # plot train loss 
    
    # text description on plot !!
    plt.title("Loss curve") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend 
   
    plt.savefig(savepath, dpi=300)

def create_ngrams(text, CONTEXT_SIZE):
    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.
    # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
    ngrams = [
        (
            [text[i - j - 1] for j in range(CONTEXT_SIZE)],
            text[i]
        )
        for i in range(CONTEXT_SIZE, len(text))
    ]

    return ngrams

def fit_embeddings(ngrams:list, vocab:set, word_to_ix:dict, model, optimiser, n_epochs:int, CONTEXT_SIZE:int, EMBEDDING_DIM:int): 
    # define loss function
    losses = []
    loss_function = nn.NLLLoss()

    for epoch in range(n_epochs):
        total_loss = 0
        for context, target in ngrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimiser.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

        losses.append(total_loss)

    return model, losses

def main(): 
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10

    # We will use Shakespeare Sonnet 2
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    ngrams = create_ngrams(test_sentence, CONTEXT_SIZE)
    
    # define vocabulary (unique words with set function)
    vocab = set(test_sentence)

    # mapping indices to words in vocab
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    # init model and optimiser
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimiser = optim.SGD(model.parameters(), lr=0.001)

    fitted_model, losses = fit_embeddings(
                    ngrams=ngrams,
                    vocab=vocab,
                    word_to_ix=word_to_ix, 
                    model=model,
                    optimiser=optimiser,
                    n_epochs=20,
                    CONTEXT_SIZE=CONTEXT_SIZE,
                    EMBEDDING_DIM=EMBEDDING_DIM
                    )
    
    # To get the embedding of a particular word, e.g. "beauty"
    print(f"Embedding for 'beauty: '{fitted_model.embeddings.weight[word_to_ix['beauty']]}")

    # plot the loss
    savepath = pathlib.Path(__file__).parents[1] / "hello.png"
    plot_loss(losses, 20, savepath)


if __name__ == "__main__":
    main()