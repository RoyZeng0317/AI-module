from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .imaging import (
    ImageEditorError, image_info, enhance_image, inpaint_image, cutout_image
)

app = typer.Typer(
    name="aihe",
    help="本機 CLI 影像處理工具：資訊檢查、影像增強、遮罩修復、主體擷取。",
    no_args_is_help=True,
)
console = Console()


@app.command()
def inspect(image: Path = typer.Argument(..., exists=True, readable=True)):
    """顯示圖片格式、尺寸、色彩模式與檔案大小。"""
    try:
        info = image_info(image)
        table = Table(title="Image information")
        table.add_column("項目")
        table.add_column("內容")
        for key, value in info.items():
            table.add_row(key, str(value))
        console.print(table)
    except ImageEditorError as exc:
        raise typer.BadParameter(str(exc))


@app.command()
def enhance(
    image: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/enhanced.png"), "--output", "-o"),
    strength: int = typer.Option(5, min=0, max=20, help="降噪強度，0 表示不降噪。"),
    sharpen: float = typer.Option(0.25, min=0.0, max=1.0, help="銳化強度。"),
):
    """輕度降噪與銳化；不改變圖片長寬。"""
    try:
        result = enhance_image(image, output, strength, sharpen)
        rprint(f"[green]完成：[/green]{result.resolve()}")
    except ImageEditorError as exc:
        raise typer.BadParameter(str(exc))


@app.command()
def inpaint(
    image: Path = typer.Argument(..., exists=True, readable=True),
    mask: Path = typer.Option(..., "--mask", "-m", exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/repaired.png"), "--output", "-o"),
    radius: float = typer.Option(5.0, min=1.0, max=20.0, help="修復半徑。"),
):
    """使用遮罩修復一般物件或背景瑕疵。"""
    try:
        result = inpaint_image(image, mask, output, radius)
        rprint(f"[green]完成：[/green]{result.resolve()}")
    except ImageEditorError as exc:
        raise typer.BadParameter(str(exc))


@app.command()
def cutout(
    image: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("outputs/subject.png"), "--output", "-o"),
):
    """使用可選 rembg 套件擷取前景並輸出透明 PNG。"""
    try:
        result = cutout_image(image, output)
        rprint(f"[green]完成：[/green]{result.resolve()}")
    except ImageEditorError as exc:
        raise typer.BadParameter(str(exc))


if __name__ == "__main__":
    app()
