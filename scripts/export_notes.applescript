-- Apple Notes Exporter version 1.1
-- By Johan Sanneblad
-- Exports all Apple Notes in folders with filename "<creation date> <title> [<id>]"
-- Emits JSONL progress lines to stdout for desktop app progress UI.
global exportFolder
global exportFolderPosix
global totalExportableNotes
global exportedNotes
global maxNotesToExport
global totalTargetNotes
global exportMode

on run argv
	set totalExportableNotes to 0
	set exportedNotes to 0
	set maxNotesToExport to 0
	set totalTargetNotes to 0
	set exportMode to "full"

	if (count of argv) > 0 then
		set outputPath to item 1 of argv as text
		do shell script "mkdir -p " & quoted form of outputPath
		set exportFolder to (POSIX file outputPath) as string
	else
		set exportFolder to (choose folder) as string
	end if

	if (count of argv) > 1 then
		set arg2 to (item 2 of argv as text)
		if arg2 is "delta" or arg2 is "full" then
			set exportMode to arg2
			if (count of argv) > 2 then
				try
					set parsedLimit to (item 3 of argv) as integer
					if parsedLimit > 0 then set maxNotesToExport to parsedLimit
				end try
			end if
		else
			try
				set parsedLimit to arg2 as integer
				if parsedLimit > 0 then set maxNotesToExport to parsedLimit
			end try
		end if
	end if

	set exportFolderPosix to POSIX path of exportFolder

	my exportAllNotes()

	if (count of argv) > 0 then
		return POSIX path of exportFolder
	end if
end run

on escapeJson(valueText)
	set escaped to my replaceText("\\", "\\\\", valueText)
	set escaped to my replaceText("\"", "\\\"", escaped)
	set escaped to my replaceText(return, " ", escaped)
	set escaped to my replaceText(linefeed, " ", escaped)
	set escaped to my replaceText(tab, " ", escaped)
	return escaped
end escapeJson

on emitProgress(kind, exportedCount, totalCount, noteName)
	set safeNote to my escapeJson(noteName as text)
	set payload to "{\"type\":\"" & kind & "\",\"phase\":\"export\",\"exported\":" & (exportedCount as text) & ",\"total\":" & (totalCount as text) & ",\"note\":\"" & safeNote & "\"}"
	do shell script "/bin/echo " & quoted form of payload
end emitProgress

on extractNoteId(filePath)
	set fileName to do shell script "basename " & quoted form of filePath
	if fileName does not contain "[" then return ""
	if fileName does not contain "]" then return ""

	set oldDelimiters to AppleScript's text item delimiters
	set AppleScript's text item delimiters to "["
	set leftParts to text items of fileName
	if (count of leftParts) < 2 then
		set AppleScript's text item delimiters to oldDelimiters
		return ""
	end if

	set tailPart to item -1 of leftParts
	set AppleScript's text item delimiters to "]"
	set rightParts to text items of tailPart
	set AppleScript's text item delimiters to oldDelimiters
	if (count of rightParts) < 1 then return ""
	return item 1 of rightParts
end extractNoteId

on lookupExistingPath(noteId, fileRecords)
	if noteId is "" then return ""
	set lookupPrefix to noteId & tab
	repeat with recordLine in fileRecords
		set lineText to recordLine as text
		if lineText starts with lookupPrefix then
			set oldDelimiters to AppleScript's text item delimiters
			set AppleScript's text item delimiters to tab
			set lineItems to text items of lineText
			set AppleScript's text item delimiters to oldDelimiters
			if (count of lineItems) > 1 then
				return item 2 of lineItems
			end if
		end if
	end repeat
	return ""
end lookupExistingPath

-- Simple text replacing
on replaceText(find, replace, subject)
	set prevTIDs to text item delimiters of AppleScript
	set text item delimiters of AppleScript to find
	set subject to text items of subject

	set text item delimiters of AppleScript to replace
	set subject to "" & subject
	set text item delimiters of AppleScript to prevTIDs

	return subject
end replaceText

-- Returns an HTML file to save the note in. We have to escape
-- the colons or AppleScript gets upset.
-- noteContainer must end with ":"
on noteNameToFilePath(noteContainer, noteName)
	global exportFolder
	set strLength to the length of noteName

	if strLength > 250 then
		set noteName to text 1 thru 250 of noteName
	end if

	set noteFolder to exportFolder & noteContainer

	-- Create folder if it does not exist
	set outputFolder to POSIX path of (noteFolder as text)
	do shell script "mkdir -p " & quoted form of outputFolder

	set fileName to (noteFolder & replaceText(":", "_", noteName) & ".html")
	return fileName
end noteNameToFilePath

on exportAllNotes()
	set exportedNotes to 0
	set totalTargetNotes to 0
	set processedNotes to 0
	set currentNoteIds to "|"
	set existingFileRecords to {}

	if exportMode is "delta" then
		set filesByIdRaw to ""
		try
			set filesByIdRaw to do shell script "find " & quoted form of exportFolderPosix & " -type f \\( -name '*.html' -o -name '*.htm' \\) | while IFS= read -r f; do b=$(basename \"$f\"); id=$(printf '%s' \"$b\" | sed -n 's/^.*\\[\\([^]]*\\)\\].*$/\\1/p'); if [ -n \"$id\" ]; then printf '%s\\t%s\\n' \"$id\" \"$f\"; fi; done"
		end try
		if filesByIdRaw is not "" then set existingFileRecords to paragraphs of filesByIdRaw
	end if

	tell application "Notes"
		set sourceNotes to notes of default account
		set exportableNotes to {}

		repeat with theNote in sourceNotes
			set noteLocked to false
			try
				set noteLocked to password protected of theNote as boolean
			end try
			if not noteLocked then set end of exportableNotes to theNote
		end repeat

		set totalExportableNotes to count of exportableNotes
		if maxNotesToExport > 0 then
			if totalExportableNotes > maxNotesToExport then
				set totalTargetNotes to maxNotesToExport
			else
				set totalTargetNotes to totalExportableNotes
			end if
		else
			set totalTargetNotes to totalExportableNotes
		end if

		my emitProgress("progress", 0, totalTargetNotes, "")

		repeat with theNote in exportableNotes
			if processedNotes is greater than or equal to totalTargetNotes then
				exit repeat
			end if

			set noteModificationDate to modification date of theNote as date
			set noteCreationDate to creation date of theNote as date

			set noteID to id of theNote as string
			set oldDelimiters to AppleScript's text item delimiters
			set AppleScript's text item delimiters to "/"
			set theArray to every text item of noteID
			set AppleScript's text item delimiters to oldDelimiters
			if length of theArray > 4 then
				-- e.g. x-coredata://.../ICNote/p599 => keep final id segment
				set noteID to item 5 of theArray
			else
				set noteID to ""
			end if
			if noteID is not "" then set currentNoteIds to currentNoteIds & noteID & "|"

			set shouldWrite to true
			set fileName to noteID
			set filepath to ""
			if exportMode is "delta" and noteID is not "" then
				set existingPath to my lookupExistingPath(noteID, existingFileRecords)
				if existingPath is not "" then
					try
						set existingAlias to (POSIX file existingPath) as alias
						tell application "Finder"
							set existingModificationDate to modification date of file existingAlias as date
						end tell
						if existingModificationDate is equal to noteModificationDate then
							set shouldWrite to false
						end if
						set filepath to existingAlias as string
						set fileName to do shell script "basename " & quoted form of existingPath
					on error
						set filepath to ""
						set shouldWrite to true
					end try
				end if
			end if

			if filepath is "" then
				set noteFolder to ""
				set theContainer to container of theNote
				try
					repeat while theContainer is not missing value
						set noteFolder to (name of theContainer) & ":" & noteFolder
						set theContainer to (container of theContainer)
					end repeat
				end try

				-- Prefix all notes by creation date (YYYYMMDD)
				set yy to (year of noteCreationDate as text)
				set mm to text -2 through -1 of ("0" & (month of noteCreationDate as integer))
				set dd to text -2 through -1 of ("0" & (day of noteCreationDate))
				set datePrefix to yy & mm & dd

				-- File name composed by creation date + note title + id
				set fileName to ((datePrefix & " " & name of theNote as string) & " [" & noteID & "]") as string
				set filepath to noteNameToFilePath(noteFolder, fileName) of me
			end if

			if shouldWrite then
				set theText to body of theNote as string
				set noteFile to open for access filepath with write permission
				set eof noteFile to 0
				write theText to noteFile as «class utf8»
				close access noteFile

				tell application "Finder"
					set modification date of file (filepath) to noteModificationDate
				end tell

				set exportedNotes to exportedNotes + 1
			end if

			set processedNotes to processedNotes + 1
			my emitProgress("progress", processedNotes, totalTargetNotes, fileName)
		end repeat
	end tell

	if exportMode is "delta" then
		set existingFilesRaw to ""
		try
			set existingFilesRaw to do shell script "find " & quoted form of exportFolderPosix & " -type f \\( -name '*.html' -o -name '*.htm' \\)"
		end try

		if existingFilesRaw is not "" then
			set existingFiles to paragraphs of existingFilesRaw
			repeat with existingFile in existingFiles
				set filePath to existingFile as text
				set existingNoteId to my extractNoteId(filePath)
				if existingNoteId is not "" then
					if currentNoteIds does not contain ("|" & existingNoteId & "|") then
						do shell script "rm -f " & quoted form of filePath
					end if
				end if
			end repeat
		end if
	end if

	my emitProgress("complete", processedNotes, totalTargetNotes, "")
end exportAllNotes
