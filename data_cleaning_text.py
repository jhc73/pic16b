import pandas as pd
import re
import unicodedata
from chardet import detect

def clean_title_text(text):
    """
    Clean title text by handling various encoding issues and garbled characters
    
    Args:
        text (str): The input text to be cleaned
        
    Returns:
        str: Cleaned text or "__CORRUPTED_ROW_DELETE__" for rows that should be deleted
    """
    if pd.isna(text) or text == '':
        return text
    
    # Convert to string type
    text = str(text)
    
    # Handle common encoding issues - attempt to fix double encoding
    try:
        # Check if it's UTF-8 incorrectly decoded as Latin-1
        if text.encode('latin-1').decode('utf-8', errors='ignore') != text:
            try:
                fixed_text = text.encode('latin-1').decode('utf-8')
                text = fixed_text
            except:
                pass
    except:
        pass
    
    # Remove or replace non-printable control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
    
    # Check if text contains mostly readable characters (Latin, numbers, common symbols)
    readable_chars = 0
    total_chars = len(text.strip())
    
    if total_chars == 0:
        return ""
    
    for char in text:
        # Check if character is Latin, number, space or common punctuation
        if (char.isascii() and (char.isalnum() or char.isspace() or char in '.,!?()[]{}|&-_:;"\'+/\\@#$%^*=~`')) or \
           unicodedata.category(char) in ['Ll', 'Lu', 'Lt', 'Nd', 'Po', 'Zs']:
            readable_chars += 1
    
    # Calculate readable character ratio
    readable_ratio = readable_chars / total_chars
    
    # 4. If text appears garbled (readable ratio < 60%), attempt to fix or mark for deletion
    if readable_ratio < 0.6:
        # Try keeping only ASCII characters
        ascii_text = ''.join(char for char in text if ord(char) < 128)
        if len(ascii_text.strip()) > 0:
            return ascii_text.strip()
        else:
            return "__CORRUPTED_ROW_DELETE__"  # Mark row for deletion
    
    # 5. Clean up excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_and_fix_encoding_issues(df):
    """
    Detect and fix encoding issues in the 'title' column of a DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame containing a 'title' column
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with encoding issues resolved
    """
    print("Starting to clean encoding issues in title column...")
    
    # Statistics before processing
    total_rows = len(df)
    problematic_rows = 0
    
    # Apply cleaning function
    cleaned_titles = []
    
    for idx, title in enumerate(df['title']):
        original_title = title
        cleaned_title = clean_title_text(title)
        
        # Check if significant changes were made
        if str(original_title) != str(cleaned_title):
            problematic_rows += 1
            if idx < 10:  # Print first 10 modified examples
                print(f"Row {idx}: '{original_title}' -> '{cleaned_title}'")
        
        cleaned_titles.append(cleaned_title)
    
    # Update DataFrame
    df['title'] = cleaned_titles
    
    # Delete rows marked as corrupted
    rows_before_deletion = len(df)
    df = df[df['title'] != "__CORRUPTED_ROW_DELETE__"].reset_index(drop=True)
    deleted_rows = rows_before_deletion - len(df)
    
    print(f"\nCleaning complete!")
    print(f"Original total rows: {total_rows}")
    print(f"Modified rows: {problematic_rows}")
    print(f"Deleted corrupted rows: {deleted_rows}")
    print(f"Final retained rows: {len(df)}")
    print(f"Deletion percentage: {deleted_rows/total_rows*100:.2f}%")
    
    return df

def main():
    """
    Main function: Read data, clean it, and save results
    """
    try:
        # Read CSV file
        print("Reading cleaned_dataset.csv...")
        
        # Try reading with different encodings
        df = None
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv('cleaned_dataset.csv', encoding=encoding)
                print(f"Successfully read file using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding} encoding: {e}")
                continue
        
        if df is None:
            raise ValueError("Failed to read file with any encoding")
        
        print(f"Original data shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        
        # Show some original data samples
        print("\nOriginal title column samples:")
        print(df['title'].head(10).to_string())
        
        # Clean the data
        df_cleaned = detect_and_fix_encoding_issues(df.copy())
        
        # Show cleaned samples
        print("\nCleaned title column samples:")
        print(df_cleaned['title'].head(10).to_string())
        
        # Save cleaned data
        output_filename = 'all_countries_videos_cleaned.csv'
        df_cleaned.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\nCleaned data saved as: {output_filename}")
        
        # Generate cleaning report
        print("\n=== Cleaning Report ===")
        
        # Check rows that still contain non-ASCII characters
        non_ascii_count = sum(1 for title in df_cleaned['title'] if not str(title).isascii())
        print(f"Titles still containing non-ASCII characters: {non_ascii_count}")
        
        # Check rows marked as potentially corrupted
        corrupted_count = sum(1 for title in df_cleaned['title'] 
                            if '[Title contains non-Latin characters - possibly corrupted]' in str(title))
        print(f"Titles marked as potentially corrupted: {corrupted_count}")
        
        # Show some examples of marked titles
        if corrupted_count > 0:
            print("\nExamples of marked titles:")
            corrupted_indices = [i for i, title in enumerate(df_cleaned['title']) 
                               if '[Title contains non-Latin characters - possibly corrupted]' in str(title)]
            for i in corrupted_indices[:5]:  # Show 5 examples
                print(f"Row {i}: Original = '{df.iloc[i]['title']}' -> Cleaned = '{df_cleaned.iloc[i]['title']}'")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Please ensure 'all_countries_videos.csv' exists in the current directory")

if __name__ == "__main__":
    main()