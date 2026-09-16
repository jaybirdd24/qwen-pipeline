package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type playerState struct {
	SchemaVersion   int               `json:"schema_version"`
	SelectedVoiceID string            `json:"selected_voice_id"`
	ActivePacks     map[string]string `json:"active_packs"`
}

type catalog struct {
	SchemaVersion int            `json:"schema_version"`
	Voices        []catalogVoice `json:"voices"`
}

type catalogVoice struct {
	VoiceID   string `json:"voice_id"`
	VoiceName string `json:"voice_name"`
	PackID    string `json:"pack_id"`
}

type manifest struct {
	PackID  string          `json:"pack_id"`
	Voice   manifestVoice   `json:"voice"`
	Stories []manifestStory `json:"stories"`
}

type manifestVoice struct {
	VoiceID string `json:"voice_id"`
}

type manifestStory struct {
	StoryID       string                   `json:"story_id"`
	TriggerObjects []string                 `json:"trigger_objects"`
	Audio          map[string]manifestAudio `json:"audio"`
	WordAudio      map[string]manifestAudio `json:"word_audio"`
}

type manifestAudio struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

func readJSON(path string, destination any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, destination); err != nil {
		return fmt.Errorf("decode %s: %w", path, err)
	}
	return nil
}

func loadState(bundle string) (playerState, error) {
	var state playerState
	err := readJSON(filepath.Join(bundle, "player-state.json"), &state)
	if err == nil && state.SchemaVersion != 1 {
		err = fmt.Errorf("unsupported player-state schema %d", state.SchemaVersion)
	}
	return state, err
}

func writeState(bundle string, state playerState) error {
	path := filepath.Join(bundle, "player-state.json")
	temporary := path + ".tmp"
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.WriteFile(temporary, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(temporary, path); err != nil {
		_ = os.Remove(temporary)
		return err
	}
	return nil
}

func safeID(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			(index > 0 && (character == '_' || character == '-')) {
			continue
		}
		return false
	}
	return true
}

func safeRelativePath(value string) bool {
	if value == "" || filepath.IsAbs(value) || filepath.Clean(value) != value {
		return false
	}
	for _, part := range strings.Split(filepath.ToSlash(value), "/") {
		if part == ".." || part == "." || part == "" {
			return false
		}
	}
	return true
}

func activeManifest(bundle string, state playerState, voiceID string) (manifest, string, error) {
	var value manifest
	packID, ok := state.ActivePacks[voiceID]
	if !ok {
		return value, "", fmt.Errorf("voice %q has no active pack", voiceID)
	}
	if !safeID(voiceID) || !safeID(packID) {
		return value, "", errors.New("player state contains an unsafe voice or pack ID")
	}
	root := filepath.Join(bundle, "packs", voiceID, packID)
	if err := readJSON(filepath.Join(root, "manifest.json"), &value); err != nil {
		return value, "", err
	}
	if value.PackID != packID || value.Voice.VoiceID != voiceID {
		return value, "", errors.New("manifest does not match its active voice and pack")
	}
	return value, root, nil
}

func resolve(bundle, trigger, language string, word bool) (string, manifestAudio, error) {
	state, err := loadState(bundle)
	if err != nil {
		return "", manifestAudio{}, err
	}
	value, root, err := activeManifest(bundle, state, state.SelectedVoiceID)
	if err != nil {
		return "", manifestAudio{}, err
	}
	for _, story := range value.Stories {
		matched := false
		for _, candidate := range story.TriggerObjects {
			if strings.EqualFold(candidate, trigger) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		audioByLanguage := story.Audio
		audioType := "story"
		if word {
			audioByLanguage = story.WordAudio
			audioType = "word"
		}
		audio, ok := audioByLanguage[language]
		if !ok {
			return "", audio, fmt.Errorf("story %q has no %q %s audio", story.StoryID, language, audioType)
		}
		if !safeRelativePath(audio.Path) {
			return "", audio, fmt.Errorf("manifest contains unsafe audio path %q", audio.Path)
		}
		path := filepath.Join(root, filepath.FromSlash(audio.Path))
		if _, err := os.Stat(path); err != nil {
			return "", audio, err
		}
		return path, audio, nil
	}
	return "", manifestAudio{}, fmt.Errorf("no story uses trigger %q", trigger)
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func verifyAudio(path string, audio manifestAudio) error {
	digest, err := fileSHA256(path)
	if err != nil {
		return err
	}
	if digest != audio.SHA256 {
		return fmt.Errorf("checksum mismatch: %s", path)
	}
	return nil
}

func listVoices(bundle string) error {
	var value catalog
	if err := readJSON(filepath.Join(bundle, "catalog.json"), &value); err != nil {
		return err
	}
	state, err := loadState(bundle)
	if err != nil {
		return err
	}
	for _, voice := range value.Voices {
		marker := " "
		if voice.VoiceID == state.SelectedVoiceID {
			marker = "*"
		}
		fmt.Printf("%s %s\t%s\t%s\n", marker, voice.VoiceID, voice.VoiceName, voice.PackID)
	}
	return nil
}

func selectVoice(bundle, voiceID string) error {
	state, err := loadState(bundle)
	if err != nil {
		return err
	}
	if _, ok := state.ActivePacks[voiceID]; !ok {
		return fmt.Errorf("voice %q is not installed", voiceID)
	}
	state.SelectedVoiceID = voiceID
	if err := writeState(bundle, state); err != nil {
		return err
	}
	fmt.Println(voiceID)
	return nil
}

func verifyBundle(bundle string) error {
	state, err := loadState(bundle)
	if err != nil {
		return err
	}
	voiceIDs := make([]string, 0, len(state.ActivePacks))
	for voiceID := range state.ActivePacks {
		voiceIDs = append(voiceIDs, voiceID)
	}
	sort.Strings(voiceIDs)
	count := 0
	for _, voiceID := range voiceIDs {
		value, root, err := activeManifest(bundle, state, voiceID)
		if err != nil {
			return err
		}
		for _, story := range value.Stories {
			for _, audioByLanguage := range []map[string]manifestAudio{story.Audio, story.WordAudio} {
				for _, audio := range audioByLanguage {
					if !safeRelativePath(audio.Path) {
						return fmt.Errorf("manifest contains unsafe audio path %q", audio.Path)
					}
					path := filepath.Join(root, filepath.FromSlash(audio.Path))
					if err := verifyAudio(path, audio); err != nil {
						return err
					}
					count++
				}
			}
		}
	}
	fmt.Printf("verified %d audio files for %d voices\n", count, len(voiceIDs))
	return nil
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: story-player -bundle PATH <list|select|resolve|play|resolve-word|play-word|verify> [arguments]")
	fmt.Fprintln(os.Stderr, "  select VOICE_ID")
	fmt.Fprintln(os.Stderr, "  resolve TRIGGER LANGUAGE")
	fmt.Fprintln(os.Stderr, "  play TRIGGER LANGUAGE")
	fmt.Fprintln(os.Stderr, "  resolve-word TRIGGER LANGUAGE")
	fmt.Fprintln(os.Stderr, "  play-word TRIGGER LANGUAGE")
}

func run() error {
	bundle := flag.String("bundle", ".", "offline Pi bundle directory")
	aplay := flag.String("aplay", "aplay", "audio player executable")
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		usage()
		return errors.New("missing command")
	}
	switch args[0] {
	case "list":
		return listVoices(*bundle)
	case "select":
		if len(args) != 2 {
			return errors.New("select requires VOICE_ID")
		}
		return selectVoice(*bundle, args[1])
	case "resolve", "play", "resolve-word", "play-word":
		if len(args) != 3 {
			return fmt.Errorf("%s requires TRIGGER and LANGUAGE", args[0])
		}
		word := args[0] == "resolve-word" || args[0] == "play-word"
		path, audio, err := resolve(*bundle, args[1], args[2], word)
		if err != nil {
			return err
		}
		if err := verifyAudio(path, audio); err != nil {
			return err
		}
		if args[0] == "resolve" || args[0] == "resolve-word" {
			fmt.Println(path)
			return nil
		}
		command := exec.Command(*aplay, path)
		command.Stdout = os.Stdout
		command.Stderr = os.Stderr
		return command.Run()
	case "verify":
		return verifyBundle(*bundle)
	default:
		usage()
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		os.Exit(1)
	}
}
