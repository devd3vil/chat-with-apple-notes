-- Apple Notes Exporter version 1.1
-- By Johan Sanneblad
-- Exports all Apple Notes in folders with filename "<creation date> <title> [<id>]"
-- Emits JSONL progress lines to stdout for desktop app progress UI.
global exportFolder
global totalExportableNotes
global exportedNotes
global maxNotesToExport
global totalTargetNotes

on run argv
	set totalExportableNotes to 0
	set exportedNotes to 0
	set maxNotesToExport to 0
	set totalTargetNotes to 0

	if (count of argv) > 0 then
		set outputPath to item 1 of argv as text
		do shell script "mkdir -p " & quoted form of outputPath
		set exportFolder to (POSIX file outputPath) as string
	else
		set exportFolder to (choose folder) as string
	end if

	if (count of argv) > 1 then
		try
			set parsedLimit to (item 2 of argv) as integer
			if parsedLimit > 0 then
				set maxNotesToExport to parsedLimit
			end if
		end try
	end if

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
			if exportedNotes is greater than or equal to totalTargetNotes then
				exit repeat
			end if

			set noteModificationDate to modification date of theNote as date
			set noteCreationDate to creation date of theNote as date

			set noteID to id of theNote as string
			set oldDelimiters to AppleScript's text item delimiters
			set AppleScript's text item delimiters to "/"
			set theArray to every text item of noteID
			set AppleScript's text item delimiters to oldDelimiters
			set noteFolder to ""

			if length of theArray > 4 then
				-- e.g. x-coredata://.../ICNote/p599 => keep final id segment
				set noteID to item 5 of theArray
			else
				set noteID to ""
			end if

			set theText to body of theNote as string
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
			set noteFile to open for access filepath with write permission

			write theText to noteFile as «class utf8»
			close access noteFile

			tell application "Finder"
				set modification date of file (filepath) to noteModificationDate
			end tell

			set exportedNotes to exportedNotes + 1
			my emitProgress("progress", exportedNotes, totalTargetNotes, fileName)

			if exportedNotes is greater than or equal to totalTargetNotes then
				exit repeat
			end if
		end repeat
	end tell

	my emitProgress("complete", exportedNotes, totalTargetNotes, "")
end exportAllNotes
