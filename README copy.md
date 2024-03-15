# Default BERT Textual Entailment Service

This application provides a default implementation of BERT Textual Entailment Service. It  is an AMF component designed to detect argument relations between propositions using BERT models. 

## Endpoints

### /bert-te

- **Methods**: GET, POST
- **Description**: 
  - **GET**: Returns information about the BERT Textual Entailment Service and its usage.
  - **POST**: Expects a file upload containing textual data. Performs textual entailment analysis using BERT models and returns the inferred argument structure.

## Usage

- Utilize the `/bert-te` endpoint to interact with the BERT Textual Entailment Service:
   - For GET requests, visit the endpoint URL to get information about the service and its usage.
   - For POST requests, upload a file containing textual data to analyze and receive the inferred argument structure.

## Input Format

- **Text File**: xAIF format input.

## Output Format

The inferred argument structure is returned in the xIAF format, containing nodes, edges, locutions, and other relevant information.

## Notes

- This service employs BERT models for textual entailment analysis, fine-tuned to detect inferences, conflicts, and non-relations between propositions.
- It can be integrated into the argument mining pipeline alongside other AMF components for further analysis and processing.
