# Absolute imports
import base64
import io
import pickle
import traceback

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import Image as IPImage, Markdown, display
from markdown import markdown
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Image, ListFlowable, ListItem, Paragraph,
                                 Preformatted, SimpleDocTemplate, Spacer)

# Typing imports
from typing import Any, Dict, List

def serialize_model(model: Any) -> str:
    """
    Serializes a trained machine learning model to a base64-encoded string.

    This function pickles the given model and encodes the binary data into a UTF-8 base64 string, 
    making it safe to store or transmit over JSON or HTTP.

    Args:
        model (Any): The trained model object to serialize. This is typically a scikit-learn-compatible model.

    Returns:
        str: A base64-encoded string representation of the pickled model.
    """
    # Serialize the model object into binary using pickle
    pickled_model = pickle.dumps(model)

    # Encode the binary into base64 and decode to UTF-8 for safe transport/storage
    b64_model = base64.b64encode(pickled_model).decode('utf-8')

    return b64_model

def deserialize_model(b64_model: str) -> Any:
    """
    Decodes and unpickles a base64-encoded machine learning model.

    Args:
        b64_model (str): A base64-encoded string representing a pickled model object.

    Returns:
        Any: The deserialized model object.

    Raises:
        ValueError: If decoding or unpickling fails.
    """
    try:
        # Decode the base64 string and unpickle the model
        pickled_model = base64.b64decode(b64_model.encode('utf-8'))
        model = pickle.loads(pickled_model)
        return model
    except Exception as e:
        raise ValueError(f"Failed to deserialize model: {e}")

def safe_serialize(obj: Any) -> Any:
    """
    Recursively serializes common data structures into JSON-safe formats.

    This utility function ensures that objects such as pandas DataFrames, Series, NumPy arrays,
    and nested dictionaries are converted into native Python types (lists, dicts, etc.)
    suitable for JSON serialization.

    Args:
        obj (Any): The input object to serialize. Can be a DataFrame, Series, ndarray, dict, or any other type.

    Returns:
        Any: A version of the input that is safe for JSON serialization.
    """
    # Convert pandas DataFrame to list of dictionaries (row-wise records)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient = "records")

    # Convert pandas Series or NumPy array to list
    elif isinstance(obj, (pd.Series, np.ndarray)):
        return obj.tolist()

    # Recursively apply serialization to dictionary values
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}

    # Return the object as-is if it doesn't match any known type
    else:
        return obj

def serialize_dataholder(dataholder: Any) -> Dict[str, Any]:
    """
    Serializes a DataHolder object into a dictionary that can be safely converted to JSON.

    Converts all internal components of the DataHolder (e.g., training/test sets, predictions, model, and scores)
    into JSON-safe formats. This includes encoding the model via base64 and converting pandas/NumPy structures
    into native Python types using `safe_serialize`.

    Args:
        dataholder (Any): A DataHolder object containing data, predictions, and model artifacts.

    Returns:
        Dict[str, Any]: A dictionary containing serialized and JSON-safe versions of the dataholder's attributes.
    """
    serialized = {
        "X_train": safe_serialize(dataholder.X_train),
        "X_test": safe_serialize(dataholder.X_test),
        "y_train": safe_serialize(dataholder.y_train),
        "y_test": safe_serialize(dataholder.y_test),
        "predictions": safe_serialize(dataholder.predictions),
        "best_model": serialize_model(dataholder.best_model),  # base64-encoded pickled model
        "target_variable": dataholder.y_train.name
    }

    # Optionally include classification metrics if they exist
    if hasattr(dataholder, "classification_scores") and dataholder.classification_scores:
        serialized["classification_scores"] = safe_serialize(dataholder.classification_scores)

    # Optionally include regression metrics if they exist
    if hasattr(dataholder, "regression_scores") and dataholder.regression_scores:
        serialized["regression_scores"] = safe_serialize(dataholder.regression_scores)

    return serialized

def figure_to_base64(fig: plt.Figure) -> str:
    """
    Converts a Matplotlib figure into a base64-encoded PNG string.

    This function is useful for embedding plots directly into HTML or JSON responses
    without saving to disk. It uses an in-memory buffer to save the figure as a PNG
    image, which is then base64-encoded.

    Args:
        fig (plt.Figure): A Matplotlib figure object to convert.

    Returns:
        str: A base64-encoded string representing the PNG image of the figure.
    """
    # Create an in-memory binary stream to store the figure
    buf = io.BytesIO()

    # Save the figure into the buffer as a PNG
    fig.savefig(buf, format = 'png', bbox_inches = 'tight', dpi = 300)

    # Move the pointer to the start of the stream
    buf.seek(0)

    # Read buffer and encode as base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64

def base64_image_to_reportlab_image(base64_str: str) -> Image:
    """
    Convert a base64-encoded PNG image string into a ReportLab Image object,
    scaled to fit within a standard letter page (with 1-inch margins).

    Args:
        base64_str (str): Base64-encoded image string (PNG format).

    Returns:
        reportlab.platypus.Image: Scaled image flowable for PDF insertion.
    """
    # Decode the base64 image
    img_data = base64.b64decode(base64_str)
    buf = io.BytesIO(img_data)

    # Open image with PIL to get dimensions
    pil_image = PILImage.open(buf)
    img_width, img_height = pil_image.size

    # Calculate scaling to fit within PDF letter page (1-inch margins)
    max_width = letter[0] - 72  # 1 inch = 72 points
    scale = min(max_width / img_width, 1.0)
    scaled_width = img_width * scale
    scaled_height = img_height * scale

    # Reset buffer for ReportLab to read the image
    buf.seek(0)
    return Image(buf, width = scaled_width, height = scaled_height)

def serialize_regression_report(anova_table: str, residual_plot: plt.Figure,
                                            judge_eval: str) -> Dict[str, Any]:
    """
    Serialize the regression report to a dictionary with base64-encoded images and structured text.

    Args:
        anova_table (str): A plain-text ANOVA table.
        residual_plot (plt.Figure): A matplotlib figure of residuals.
        judge_eval (str): Markdown-formatted evaluation content.

    Returns:
        Dict[str, Any]: Dictionary containing the structured report.
    """
    report = {
        "disclaimer": (
            "This model and explanation were generated by ModelBot, an agent designed to help non-technical "
            "users perform basic machine learning modeling, powered by Llama 3. It is not a replacement for a "
            "human data scientist, and there may be discrepancies and inaccuracies within this report."
        ),
        "anova_table": anova_table,
        "residual_plot_base64": figure_to_base64(residual_plot),
        "evaluation": judge_eval
    }
    return report   

def serialize_classification_report(confusion_matrices: plt.Figure, roc_curves: plt.Figure, 
                                    shap_values: plt.Figure, judge_eval: str) -> Dict[str, Any]:
    """
    Serialize the classification report to a dictionary with base64-encoded images and structured text.
    
    Args:
        confusion_matrices (plt.Figure): Matplotlib figure for the confusion matrix.
        roc_curves (plt.Figure): Matplotlib figure for the ROC curves.
        shap_values (plt.Figure): Matplotlib figure for SHAP values.
        judge_eval (str): Markdown-formatted evaluation/analysis text.
    
    Returns:
        Dict[str, Any]: A dictionary containing serialized report data, including base64 images.
    """
    report = {}

    # Disclaimer text
    disclaimer_text = (
        "This model and explanation were generated by ModelBot, an agent designed to help non-technical "
        "users perform basic machine learning modeling, powered by Llama 3. It is not "
        "a replacement for a human data scientist, and there may be discrepancies and "
        "inaccuracies within this report."
    )
    report["disclaimer"] = disclaimer_text

    # Convert figures to base64 (confusion matrix, ROC, SHAP)
    report["confusion_matrices_base64"] = figure_to_base64(confusion_matrices)
    report["roc_curves_base64"] = figure_to_base64(roc_curves)
    report["shap_values_base64"] = figure_to_base64(shap_values)

    # Evaluation section (Markdown format)
    report["evaluation"] = judge_eval
    
    return report

def get_custom_styles() -> Dict[str, ParagraphStyle]:
    """
    Returns a dictionary of custom styles for the report.
    
    Returns:
        Dict[str, ParagraphStyle]: Custom styles for headers, paragraphs, lists, and code.
    """
    styles = getSampleStyleSheet()
    custom_styles = {
        "h1": ParagraphStyle("Heading1", parent = styles["Heading1"], fontSize = 16, spaceAfter = 10),
        "h2": ParagraphStyle("Heading2", parent = styles["Heading2"], fontSize = 14, spaceAfter = 8),
        "h3": ParagraphStyle("Heading3", parent = styles["Heading3"], fontSize = 12, spaceAfter = 6),
        "h4": ParagraphStyle("Heading4", parent = styles["Heading4"], fontSize = 11, spaceAfter = 5),
        "h5": ParagraphStyle("Heading5", parent = styles["Normal"], fontSize = 10,
                             spaceAfter = 4, fontName = "Helvetica-Bold"),
        "p": styles["Normal"],
        "ul": styles["Normal"],
        "code": ParagraphStyle("Code", fontName = "Courier", fontSize = 7.5,
                               leading = 9, spaceAfter = 6, leftIndent = 6, rightIndent = 6)
    }
    return custom_styles

def add_disclaimer(elements: List, disclaimer_text: str, styles: dict) -> None:
    """
    Adds a disclaimer text to the list of elements for a report.

    This function appends a styled paragraph with the disclaimer text 
    and some spacing to the provided list of elements. It uses 
    ReportLab styles to format the disclaimer text.

    Args:
        elements (List): A list of elements (e.g., paragraphs, images, etc.)
                         to which the disclaimer will be added.
        disclaimer_text (str): The disclaimer text that will be added to the report.
        styles (dict): A dictionary containing ReportLab style sheets for formatting the text.
    """
    # Ensure that the provided styles are from the default sample styles.
    styles = getSampleStyleSheet()

    # Define a custom style for the disclaimer text, setting font size and color
    disclaimer_style = ParagraphStyle(
        "Disclaimer", 
        parent = styles["Normal"],  # Inherits from the default "Normal" style
        fontSize = 8,                # Small font size for the disclaimer
        textColor = "gray",          # Gray color for the disclaimer text
        spaceAfter = 2               # Small space after the disclaimer text
    )

    # Add the disclaimer text as a Paragraph element
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 12))  # Adds a 12-point vertical space after the disclaimer

    return

def add_base64_image_section(elements: List, base64_data: str, 
                             custom_styles: Dict[str, ParagraphStyle]) -> None:
    """
    Adds a base64-encoded image as a section to the report.

    This function converts the base64-encoded image data into a format 
    that can be inserted into a ReportLab document, and appends it to 
    the list of report elements. It also adds some space after the image.

    Args:
        elements (List): A list of elements (e.g., paragraphs, images, etc.) to which 
                         the image will be appended.
        base64_data (str): The base64-encoded image data that will be converted into 
                           a ReportLab image and added to the report.
        custom_styles (Dict[str, ParagraphStyle]): A dictionary of custom styles 
                                                   to format the image section.
    """
    # Convert the base64-encoded image data into a ReportLab image object
    img = base64_image_to_reportlab_image(base64_data)

    # Append the image to the report elements list
    elements.append(img)
    elements.append(Spacer(1, 12))  # Adds a 12-point vertical space after the image

    return

def add_markdown_content(elements: List, markdown_content: str, custom_styles: Dict[str, ParagraphStyle]) -> None:
    """
    Parses and adds Markdown content (headers, paragraphs, lists, tables) to the report.

    Args:
        elements (List): The list of elements to append the parsed content to.
        markdown_content (str): The Markdown-formatted content to parse and add.
        custom_styles (Dict[str, ParagraphStyle]): Custom styles for formatting different Markdown elements 
                                                   (e.g., headers, paragraphs, lists).
    """
    # Convert the Markdown content into HTML
    html = markdown(markdown_content, extensions = ["tables"])
    
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Iterate over each tag in the parsed HTML and handle different elements
    for tag in soup:
        # Handle headers (h1 to h5)
        if tag.name in ["h1", "h2", "h3", "h4", "h5"]:
            elements.append(Paragraph(tag.get_text(), custom_styles[tag.name]))
            elements.append(Spacer(1, 6))  # Add space after headers
        # Handle paragraphs
        elif tag.name == "p":
            elements.append(Paragraph(tag.get_text(), custom_styles["p"]))
            elements.append(Spacer(1, 6))  # Add space after paragraphs
        # Handle unordered lists (ul)
        elif tag.name == "ul":
            bullets = [ListItem(Paragraph(li.get_text(), custom_styles["ul"])) for li in tag.find_all("li")]
            elements.append(ListFlowable(bullets, bulletType = "bullet"))
            elements.append(Spacer(1, 6))  # Add space after list
        # Handle tables
        elif tag.name == "table":
            rows = []
            for row in tag.find_all("tr"):
                cells = [cell.get_text(strip = True) for cell in row.find_all(["th", "td"])]
                rows.append(cells)

            table = Table(rows, hAlign = "LEFT")
            table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                       ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                       ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
            elements.append(table)
            elements.append(Spacer(1, 12))  # Add space after table

    return

def serialized_regression_report_to_pdf(report: Dict[str, Any], output_path: str) -> bool:
    """
    Convert a serialized regression report dictionary back into a formatted PDF file.
    
    Args:
        report (Dict[str, Any]): A dictionary containing structured report data with keys:
            - "disclaimer" (str)
            - "anova_table" (str)
            - "residual_plot_base64" (str): Base64-encoded PNG image of the residual plot
            - "evaluation" (list): Structured list of markdown-converted content
        output_path (str): Path where the generated PDF will be saved.
    
    Returns:
        bool: True if the PDF was generated successfully, False otherwise.
    """
    try:
        custom_styles = get_custom_styles()  # Use the shared custom styles function

        elements = []

        # Add disclaimer
        disclaimer_text = report.get("disclaimer", "")
        add_disclaimer(elements, disclaimer_text, custom_styles)

        # Add ANOVA table
        if "anova_table" in report:
            anova_text = report["anova_table"]
            elements.append(Preformatted(anova_text, custom_styles["code"]))
            elements.append(Spacer(1, 12))

        # Add residual plot (base64 image)
        if "residual_plot_base64" in report:
            add_base64_image_section(elements, report["residual_plot_base64"], custom_styles)

        # Add evaluation section (Markdown)
        if "evaluation" in report:
            add_markdown_content(elements, report["evaluation"], custom_styles)

        # Build PDF
        doc = SimpleDocTemplate(output_path, pagesize = letter)
        doc.build(elements)
        return True
        
    except Exception as e:
        print("Error generating PDF from serialized regression report:", e)
        traceback.print_exc()
        return False

def serialized_classification_report_to_pdf(report: Dict[str, Any], output_path: str) -> bool:
    """
    Convert a serialized classification report dictionary back into a formatted PDF file.
    
    Args:
        report (Dict[str, Any]): A dictionary containing structured report data with keys:
            - "disclaimer" (str)
            - "confusion_matrices_base64" (str): Base64-encoded PNG image of the confusion matrix
            - "roc_curves_base64" (str): Base64-encoded PNG image of the ROC curves
            - "shap_values_base64" (str): Base64-encoded PNG image of the SHAP values
            - "evaluation" (str): Markdown-formatted analysis or commentary
        output_path (str): Path where the generated PDF will be saved.
    
    Returns:
        bool: True if the PDF was generated successfully, False otherwise.
    """
    try:
        custom_styles = get_custom_styles()  # Use the shared custom styles function

        elements = []
        
        # Add disclaimer
        disclaimer_text = report.get("disclaimer", "")
        add_disclaimer(elements, disclaimer_text, custom_styles)

        # Add base64 images (confusion matrices, ROC curves, SHAP values)
        if "confusion_matrices_base64" in report:
            add_base64_image_section(elements, report["confusion_matrices_base64"], custom_styles)
        if "roc_curves_base64" in report:
            add_base64_image_section(elements, report["roc_curves_base64"], custom_styles)
        if "shap_values_base64" in report:
            add_base64_image_section(elements, report["shap_values_base64"], custom_styles)

        # Add evaluation section (Markdown)
        if "evaluation" in report:
            add_markdown_content(elements, report["evaluation"], custom_styles)

        # Build PDF
        doc = SimpleDocTemplate(output_path, pagesize = letter)
        doc.build(elements)
        return True
        
    except Exception as e:
        print("Error generating PDF from serialized classification report:", e)
        traceback.print_exc()
        return False

def display_serialized_classification_report(report: Dict[str, str]) -> None:
    """
    Display a serialized classification report inline in a Jupyter notebook.

    Args:
        report (dict): A dictionary containing the serialized classification report with the following keys:
            - "disclaimer" (str): A disclaimer text to display.
            - "confusion_matrices_base64" (str): Base64-encoded confusion matrix image.
            - "roc_curves_base64" (str): Base64-encoded ROC curves image.
            - "shap_values_base64" (str): Base64-encoded SHAP values image.
            - "evaluation" (str): Markdown-formatted text containing the evaluation report.

    Returns:
        None: This function displays content directly in the Jupyter notebook and does not return any value.
    """
    
    # Display disclaimer if present in the report
    if "disclaimer" in report:
        display(Markdown(f"**Disclaimer:** {report['disclaimer']}"))

    # Helper function to display base64 images with a title
    def show_image(title: str, b64_data: str) -> None:
        """
        Display a base64-encoded image with a given title in the Jupyter notebook.

        Args:
            title (str): The title for the image section.
            b64_data (str): Base64-encoded image data to display.
        
        Returns:
            None: The function displays the image inline in the notebook.
        """
        display(Markdown(f"### {title}"))
        display(IPImage(data = base64.b64decode(b64_data)))

    # Display confusion matrix image if available
    if "confusion_matrices_base64" in report:
        show_image("Confusion Matrix", report["confusion_matrices_base64"])

    # Display ROC curve image if available
    if "roc_curves_base64" in report:
        show_image("ROC Curves", report["roc_curves_base64"])

    # Display SHAP values image if available
    if "shap_values_base64" in report:
        show_image("SHAP Values", report["shap_values_base64"])

    # Display the markdown-formatted evaluation text if available
    if "evaluation" in report:
        display(Markdown(report["evaluation"]))

    return

def display_serialized_regression_report(report: Dict[str, str]) -> None:
    """
    Display a serialized regression report inline in a Jupyter notebook.

    Args:
        report (dict): A dictionary containing the serialized regression report with the following keys:
            - "disclaimer" (str): A disclaimer text to display.
            - "anova_table" (str): The ANOVA table to display, formatted as a string.
            - "residual_plot_base64" (str): Base64-encoded residual plot image.
            - "evaluation" (str): Markdown-formatted text containing the evaluation report.

    Returns:
        None: This function displays content directly in the Jupyter notebook and does not return any value.
    """
    
    # Display disclaimer if present in the report
    if "disclaimer" in report:
        display(Markdown(f"**Disclaimer:** {report['disclaimer']}"))

    # Display ANOVA table if present in the report
    if "anova_table" in report:
        display(Markdown("### ANOVA Table"))
        display(Markdown(f"```\n{report['anova_table']}\n```"))

    # Display residual plot image if available
    if "residual_plot_base64" in report:
        display(Markdown("### Residual Plot"))
        display(IPImage(data = base64.b64decode(report["residual_plot_base64"])))

    # Display the markdown-formatted evaluation text if available
    if "evaluation" in report:
        display(Markdown(report["evaluation"]))

    return