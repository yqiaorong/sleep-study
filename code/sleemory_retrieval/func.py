def plot2D(data, size, yrange, xrange, ylabel, xlabel, save_path):
    """data: (y, x)"""
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=size)
    im = plt.imshow(data, cmap='viridis',
                    extent=[xrange[0], xrange[1], yrange[0], yrange[1]], 
                    origin='lower', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('Values')

    # Plot borders
    # plt.plot([0, 2.5], [0,0], 'k--', lw=0.4)
    # plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Encoding accuracies')
    fig.tight_layout()
    plt.savefig(save_path)     
    
    