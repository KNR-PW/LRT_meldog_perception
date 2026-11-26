#include "meldog_perception/tracking/byte_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "meldog_perception/tracking/linear_assignment.hpp"

namespace meldog_perception::tracking {

    namespace {

        float iou(const cv::Rect2f& a, const cv::Rect2f& b)
        {
            const float intersection = (a & b).area();
            const float unionArea = a.area() + b.area() - intersection;
            return unionArea > 0.0F ? intersection / unionArea : 0.0F;
        }

        bool isValid(const cv::Rect2f& box)
        {
            return std::isfinite(box.x) && std::isfinite(box.y) && box.width > 1.0F && box.height > 1.0F;
        }
    }

    ByteTracker::TrackData::TrackData(const Detection& detection, int index, double timestamp)
        : filter(detection.box),
          score(detection.score),
          classId(detection.classId),
          detectionIndex(index),
          firstSeen(timestamp),
          lastSeen(timestamp)
    {
    }

    ByteTracker::ByteTracker(TrackerConfig config)
        : _config(config)
    {
    }

    const std::vector<Track>& ByteTracker::update(const std::vector<Detection>& detections, double timestamp)
    {
        _events.clear();

        if (_hasTimestamp && timestamp < _lastTimestamp) {
            for (auto& track : _tracks) {
                remove(track, timestamp);
            }
            reset();
        }

        const double steps = _hasTimestamp ? (timestamp - _lastTimestamp) * _config.referenceFps : 1.0;
        _hasTimestamp = true;
        _lastTimestamp = timestamp;

        for (auto& track : _tracks) {
            track.detectionIndex = -1;
            track.filter.predict(steps, track.state == TrackState::Lost);
        }

        std::vector<size_t> high;
        std::vector<size_t> low;
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& detection = detections[i];
            if (!isValid(detection.box)) {
                continue;
            }
            if (detection.score >= _config.highThreshold) {
                high.push_back(i);
            } else if (detection.score >= _config.lowThreshold) {
                low.push_back(i);
            }
        }

        std::vector<size_t> confirmed;
        std::vector<size_t> tentative;
        for (size_t i = 0; i < _tracks.size(); ++i) {
            (_tracks[i].state == TrackState::Tentative ? tentative : confirmed).push_back(i);
        }

        // 1. Confirmed and lost tracks vs high-score detections.
        const Association first =
            associate(confirmed, high, detections, _config.matchThreshold, _config.fuseScore);
        for (const auto& [track, detection] : first.matches) {
            applyMatch(_tracks[track], detections, detection, timestamp);
        }

        // 2. Tracks still visible in the last frame vs low-score detections.
        std::vector<size_t> remaining;
        for (const size_t track : first.unmatchedTracks) {
            if (_tracks[track].state == TrackState::Tracked) {
                remaining.push_back(track);
            }
        }
        const Association second =
            associate(remaining, low, detections, _config.secondMatchThreshold, false);
        for (const auto& [track, detection] : second.matches) {
            applyMatch(_tracks[track], detections, detection, timestamp);
        }
        for (const size_t track : second.unmatchedTracks) {
            _tracks[track].state = TrackState::Lost;
        }

        // 3. Tracks started in the previous frame vs the remaining high-score detections.
        const Association third = associate(tentative, first.unmatchedDetections, detections,
                                            _config.unconfirmedMatchThreshold, _config.fuseScore);
        for (const auto& [track, detection] : third.matches) {
            applyMatch(_tracks[track], detections, detection, timestamp);
            confirm(_tracks[track], timestamp);
        }
        for (const size_t track : third.unmatchedTracks) {
            remove(_tracks[track], timestamp);
        }

        // 4. New tracks from what is left.
        for (const size_t detection : third.unmatchedDetections) {
            if (detections[detection].score < _config.newTrackThreshold) {
                continue;
            }
            _tracks.emplace_back(detections[detection], static_cast<int>(detection), timestamp);
            if (_firstFrame) {
                confirm(_tracks.back(), timestamp);
            }
        }
        _firstFrame = false;

        // 5. People gone for too long.
        for (auto& track : _tracks) {
            if (track.state == TrackState::Lost && timestamp - track.lastSeen > _config.lostTimeout) {
                remove(track, timestamp);
            }
        }

        removeDuplicates(timestamp);

        _tracks.erase(std::remove_if(_tracks.begin(), _tracks.end(),
                                     [](const TrackData& track) { return track.state == TrackState::Removed; }),
                      _tracks.end());
        buildOutput();
        return _output;
    }

    size_t ByteTracker::lostCount() const
    {
        return static_cast<size_t>(std::count_if(_tracks.begin(), _tracks.end(),
            [](const TrackData& track) { return track.state == TrackState::Lost; }));
    }

    void ByteTracker::reset()
    {
        _tracks.clear();
        _output.clear();
        _hasTimestamp = false;
        _lastTimestamp = 0.0;
        _firstFrame = true;
    }

    ByteTracker::Association ByteTracker::associate(const std::vector<size_t>& trackIndices,
                                                    const std::vector<size_t>& detectionIndices,
                                                    const std::vector<Detection>& detections, float threshold,
                                                    bool fuseScore) const
    {
        cv::Mat1d cost(static_cast<int>(trackIndices.size()), static_cast<int>(detectionIndices.size()));
        for (int r = 0; r < cost.rows; ++r) {
            const cv::Rect2f trackBox = _tracks[trackIndices[r]].filter.box();
            for (int c = 0; c < cost.cols; ++c) {
                const Detection& detection = detections[detectionIndices[c]];
                double similarity = iou(trackBox, detection.box);
                if (fuseScore) {
                    similarity *= detection.score;
                }
                cost(r, c) = 1.0 - similarity;
            }
        }

        const AssignmentResult assignment = solveAssignment(cost, threshold);

        Association result;
        for (const auto& [r, c] : assignment.matches) {
            result.matches.emplace_back(trackIndices[r], detectionIndices[c]);
        }
        for (const int r : assignment.unmatchedRows) {
            result.unmatchedTracks.push_back(trackIndices[r]);
        }
        for (const int c : assignment.unmatchedCols) {
            result.unmatchedDetections.push_back(detectionIndices[c]);
        }
        return result;
    }

    void ByteTracker::applyMatch(TrackData& track, const std::vector<Detection>& detections, size_t index,
                                 double timestamp)
    {
        const Detection& detection = detections[index];
        track.filter.update(detection.box);
        track.score = detection.score;
        track.classId = detection.classId;
        track.detectionIndex = static_cast<int>(index);
        track.lastSeen = timestamp;
        ++track.hits;
        if (track.state == TrackState::Lost) {
            track.state = TrackState::Tracked;
        }
    }

    void ByteTracker::confirm(TrackData& track, double timestamp)
    {
        track.state = TrackState::Tracked;
        track.id = _nextId++;
        _events.push_back({TrackEvent::Type::Entered, track.id, timestamp});
    }

    void ByteTracker::remove(TrackData& track, double timestamp)
    {
        if (track.state == TrackState::Removed) {
            return;
        }
        if (track.id > 0) {
            _events.push_back({TrackEvent::Type::Left, track.id, timestamp});
        }
        track.state = TrackState::Removed;
    }

    void ByteTracker::removeDuplicates(double timestamp)
    {
        // A lost track and a tracked one on the same person: keep the older one
        // (it has the more established ID), as ByteTrack does.
        for (auto& tracked : _tracks) {
            if (tracked.state != TrackState::Tracked) {
                continue;
            }
            for (auto& lost : _tracks) {
                if (lost.state != TrackState::Lost) {
                    continue;
                }
                if (iou(tracked.filter.box(), lost.filter.box()) <= _config.duplicateIou) {
                    continue;
                }
                if (tracked.firstSeen <= lost.firstSeen) {
                    remove(lost, timestamp);
                } else {
                    remove(tracked, timestamp);
                    break;
                }
            }
        }
    }

    void ByteTracker::buildOutput()
    {
        _output.clear();
        for (const auto& track : _tracks) {
            if (track.state != TrackState::Tracked) {
                continue;
            }
            Track out;
            out.id = track.id;
            out.box = track.filter.box();
            out.velocity = track.filter.velocity() * static_cast<float>(_config.referenceFps);
            out.score = track.score;
            out.classId = track.classId;
            out.detectionIndex = track.detectionIndex;
            out.firstSeen = track.firstSeen;
            out.lastSeen = track.lastSeen;
            out.hits = track.hits;
            _output.push_back(out);
        }
        std::sort(_output.begin(), _output.end(), [](const Track& a, const Track& b) { return a.id < b.id; });
    }
}
