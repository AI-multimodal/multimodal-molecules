#!/usr/bin/bash

header_lower_word=3
upper_word=4

for file in *; do

    # If self, then continue
    if [[ "$file" == "postprocess.sh" ]]; then
        continue
    fi

    # If worst/best already...
    if [[ "$file" == *"worst"* ]]; then
        continue
    fi
    if [[ "$file" == *"best"* ]]; then
        continue
    fi

    file_basename=$(basename -s .tex "$file")

    # Remove the one line that has FG in it, since that's saved as an index
    # by pandas
    grep -v "FG &" "$file" > tmpfile && mv tmpfile "$file"
    grep -v "toprule" "$file" > tmpfile && mv tmpfile "$file"
    grep -v "bottomrule" "$file" > tmpfile && mv tmpfile "$file"

    # Get the core pieces of the table and put them in a temp file
    n_current_lines=$(wc -l "$file" | awk '{ print $1 }')
    lower_word=$((n_current_lines-1))
    sed -n "$upper_word,${lower_word}p" "$file" > tmpfile

    # Get the header
    head -n "$header_lower_word" "$file" > header

    # Get the footer
    tail -n 1 "$file" > footer

    # Get the total number of lines in the temporary file
    temp_file_n_current_lines=$(wc -l tmpfile | awk '{ print $1 }')
    second_file_lines=$((temp_file_n_current_lines / 2))

    first_file_lines=$((temp_file_n_current_lines-second_file_lines))

    # Get the cores of the two files
    head -n "$first_file_lines" tmpfile > core_upper
    tail -n "$second_file_lines" tmpfile > core_lower

    # Stitch everything together
    file_upper="$file_basename"_best.tex
    cat header > "$file_upper"
    cat core_upper >> "$file_upper"
    cat footer >> "$file_upper"

    file_lower="$file_basename"_worst.tex
    cat header > "$file_lower"
    cat core_lower >> "$file_lower"
    cat footer >> "$file_lower"

    # Remove temp files
    rm tmpfile
    rm header
    rm footer
    rm core_upper
    rm core_lower

done
