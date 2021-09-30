import click
import style_transfer.style_transfer as st

@click.command()
@click.option("--source", default="images/hockney.jpg", help="Source image to copy the style")
@click.option("--target", default="images/octopus.jpg", help="Target image to apply the style")
@click.option("--steps", default=2000, help="Total steps to train")
@click.option("--show_every", default=200, help="When to show the current state of the output image")
@click.option("--learning_rate", default=0.003, help="Learning rate")
def main(source, target, steps, show_every, learning_rate):
    st.transfer_style(source, target, steps, show_every, learning_rate)
    
if __name__ == "__main__":
    main()
    