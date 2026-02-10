-- Apple Notes Exporter version 1.1
-- By Johan Sanneblad
-- Exports all Apple Notes in folders with filename "<creation date> <title> [<id>]"
-- Emits JSONL progress lines to stderr for desktop app progress UI.
global exportFolder
global exportFolderPosix
global totalExportableNotes
global exportedNotes
global maxNotesToExport
global totalTargetNotes
global exportMode
global deltaSinceEpochSeconds

on run argv
	set totalExportableNotes to 0
	set exportedNotes to 0
	set maxNotesToExport to 0
	set totalTargetNotes to 0
	set exportMode to "full"
	set deltaSinceEpochSeconds to 0

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
				set arg3 to (item 3 of argv as text)
				try
					set parsedArg3 to arg3 as integer
					if exportMode is "delta" and parsedArg3 > 1000000000 then
						set deltaSinceEpochSeconds to parsedArg3
					else if parsedArg3 > 0 then
						set maxNotesToExport to parsedArg3
					end if
				end try
			end if
			if (count of argv) > 3 then
				try
					set parsedLimit to (item 4 of argv) as integer
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
	log payload
end emitProgress

on lookupExistingRecord(noteId, fileRecords)
	if noteId is "" then return {"", 0}
	set lookupPrefix to noteId & tab
	repeat with recordLine in fileRecords
		set lineText to recordLine as text
		if lineText starts with lookupPrefix then
			set oldDelimiters to AppleScript's text item delimiters
			set AppleScript's text item delimiters to tab
			set lineItems to text items of lineText
			set AppleScript's text item delimiters to oldDelimiters
			if (count of lineItems) > 2 then
				set parsedMtime to 0
				try
					set parsedMtime to (item 2 of lineItems) as real
				end try
				return {item 3 of lineItems, parsedMtime}
			else if (count of lineItems) is 2 then
				return {item 2 of lineItems, 0}
			end if
		end if
	end repeat
	return {"", 0}
end lookupExistingRecord

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

on normalizeNoteId(rawId)
	set noteID to rawId as string
	if noteID is "" then return ""
	set oldDelimiters to AppleScript's text item delimiters
	set AppleScript's text item delimiters to "/"
	set idParts to text items of noteID
	set AppleScript's text item delimiters to oldDelimiters
	if (count of idParts) > 4 then
		return item 5 of idParts
	end if
	return noteID
end normalizeNoteId

on exportAllNotes()
	set exportedNotes to 0
	set totalTargetNotes to 0
	set processedNotes to 0
	set skippedNotes to 0
	set currentNoteIds to "|"
	set existingFileRecords to {}
	set unixEpochDate to date "Thursday, January 1, 1970 at 12:00:00 AM"
	set hasPrecomputedCurrentIds to false
	set canCleanupRemovedFiles to false

	if exportMode is "delta" then
		if deltaSinceEpochSeconds is 0 then set canCleanupRemovedFiles to true
		set filesByIdRaw to ""
		try
			set filesByIdRaw to do shell script "find " & quoted form of exportFolderPosix & " -type f \\( -name '*.html' -o -name '*.htm' \\) | while IFS= read -r f; do b=$(basename \"$f\"); id=$(printf '%s' \"$b\" | sed -n 's/^.*\\[\\([^]]*\\)\\].*$/\\1/p'); if [ -n \"$id\" ]; then m=$(stat -f %m \"$f\" 2>/dev/null || echo 0); printf '%s\\t%s\\t%s\\n' \"$id\" \"$m\" \"$f\"; fi; done"
		end try
		if filesByIdRaw is not "" then set existingFileRecords to paragraphs of filesByIdRaw
	end if

	tell application "Notes"
		set allNotes to notes of default account
		set totalExportableNotes to count allNotes
		set sourceNotes to allNotes

		if exportMode is "delta" and deltaSinceEpochSeconds > 0 then
			try
				set cutoffDate to unixEpochDate + deltaSinceEpochSeconds
				set sourceNotes to notes of default account whose modification date is greater than cutoffDate
			on error
				set sourceNotes to allNotes
			end try

			try
				set allRawIds to id of notes of default account
				repeat with rawId in allRawIds
					set normalizedId to my normalizeNoteId(rawId as string)
					if normalizedId is not "" then set currentNoteIds to currentNoteIds & normalizedId & "|"
				end repeat
				set hasPrecomputedCurrentIds to true
				set canCleanupRemovedFiles to true
			end try
		end if

		set totalCandidateNotes to count sourceNotes
		if maxNotesToExport > 0 then
			if totalCandidateNotes > maxNotesToExport then
				set totalTargetNotes to maxNotesToExport
			else
				set totalTargetNotes to totalCandidateNotes
			end if
		else
			set totalTargetNotes to totalCandidateNotes
		end if

		my emitProgress("progress", 0, totalTargetNotes, "")

		repeat with theNote in sourceNotes
			if processedNotes is greater than or equal to totalTargetNotes then
				exit repeat
			end if

			set skipCurrentNote to false
			set noteModificationDate to missing value
			set noteID to ""
			set noteEpoch to 0
			try
				set noteModificationDate to modification date of theNote as date
				set noteID to id of theNote as string
			on error errMsg
				set processedNotes to processedNotes + 1
				set skippedNotes to skippedNotes + 1
				my emitProgress("progress", processedNotes, totalTargetNotes, "Skipped note (metadata timeout/error): " & errMsg)
				set skipCurrentNote to true
			end try

			if not skipCurrentNote then
				set noteID to my normalizeNoteId(noteID)
				if noteID is not "" and (not hasPrecomputedCurrentIds) then
					set currentNoteIds to currentNoteIds & noteID & "|"
				end if

				set shouldWrite to true
				set fileName to noteID
				set filepath to ""
				set existingPath to ""
				set existingMtime to 0
				set noteEpoch to (noteModificationDate - unixEpochDate)
				if exportMode is "delta" and noteID is not "" then
					set existingInfo to my lookupExistingRecord(noteID, existingFileRecords)
					set existingPath to item 1 of existingInfo
					set existingMtime to item 2 of existingInfo
					if existingPath is not "" then
						if noteEpoch is less than or equal to (existingMtime + 1) then
							set shouldWrite to false
						end if
						set filepath to (POSIX file existingPath) as string
						set fileName to do shell script "basename " & quoted form of existingPath
					end if
				end if

				if shouldWrite and filepath is "" then
					set noteFolder to ""
					set noteCreationDate to missing value
					set theContainer to missing value
					try
						set noteCreationDate to creation date of theNote as date
						set theContainer to container of theNote
					on error errMsg
						set processedNotes to processedNotes + 1
						set skippedNotes to skippedNotes + 1
						my emitProgress("progress", processedNotes, totalTargetNotes, "Skipped note (container/creation error): " & errMsg)
						set skipCurrentNote to true
					end try
					if not skipCurrentNote then
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
				end if

				if shouldWrite and not skipCurrentNote then
					set theText to ""
					try
						with timeout of 20 seconds
							set theText to body of theNote as string
						end timeout
					on error errMsg
						set processedNotes to processedNotes + 1
						set skippedNotes to skippedNotes + 1
						my emitProgress("progress", processedNotes, totalTargetNotes, "Skipped note (body error): " & errMsg)
						set skipCurrentNote to true
					end try

					if not skipCurrentNote then
						set noteFile to open for access filepath with write permission
						set eof noteFile to 0
						write theText to noteFile as «class utf8»
						close access noteFile

						tell application "Finder"
							set modification date of file (filepath) to noteModificationDate
						end tell

						set exportedNotes to exportedNotes + 1
					end if
				end if

				if not skipCurrentNote then
					set processedNotes to processedNotes + 1
					set shouldEmitProgress to true
					if exportMode is "delta" and (not shouldWrite) then
						set shouldEmitProgress to ((processedNotes mod 10) is 0) or (processedNotes is totalTargetNotes)
					end if
					if shouldEmitProgress then
						my emitProgress("progress", processedNotes, totalTargetNotes, fileName)
					end if
				end if
			end if
		end repeat
	end tell

	if exportMode is "delta" and canCleanupRemovedFiles then
		repeat with recordLine in existingFileRecords
			set lineText to recordLine as text
			set oldDelimiters to AppleScript's text item delimiters
			set AppleScript's text item delimiters to tab
			set lineItems to text items of lineText
			set AppleScript's text item delimiters to oldDelimiters
			if (count of lineItems) > 2 then
				set existingNoteId to item 1 of lineItems
				set filePath to item 3 of lineItems
				if existingNoteId is not "" then
					if currentNoteIds does not contain ("|" & existingNoteId & "|") then
						do shell script "rm -f " & quoted form of filePath
					end if
				end if
			end if
		end repeat
	else if exportMode is "delta" then
		log ("{\"type\":\"warning\",\"phase\":\"export\",\"message\":\"Skipped removal cleanup because current note IDs could not be resolved.\"}")
	end if

	if skippedNotes > 0 then
		log ("{\"type\":\"warning\",\"phase\":\"export\",\"message\":\"Skipped " & (skippedNotes as text) & " note(s) due to access/timeouts.\"}")
	end if
	my emitProgress("complete", processedNotes, totalTargetNotes, "")
end exportAllNotes
