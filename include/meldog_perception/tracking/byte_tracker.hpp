#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "meldog_perception/detection.hpp"
#include "meldog_perception/tracking/kalman_box_filter.hpp"

namespace meldog_perception::tracking {

    // The three knobs of the original ByteTrack; everything else is fixed in byte_tracker.cpp.
    struct TrackerConfig
    {
        // Detections with score >= highThreshold go to the main association and can
        // start new people; lower ones (down to 0.1) only keep existing tracks alive,
        // because partly occluded people often get a low score.
        float highThreshold = 0.5F;
        // Maximum matching cost, 1 - IoU * score.
        float matchThreshold = 0.8F;
        // Seconds a person may stay hidden and still get the same ID back.
        double lostTimeout = 1.0;
    };

    struct Track
    {
        int id = 0;                 // stable, starts at 1, never reused
        cv::Rect2f box;             // Kalman-filtered box, pixels
        cv::Point2f velocity;       // box center, pixels per second
        float score = 0.0F;         // score of the last matched detection
        int detectionIndex = -1;    // index in the detections given to the last update()
        double lastSeen = 0.0;      // seconds
    };

    struct TrackEvent
    {
        enum class Type { Entered, Left };

        Type type = Type::Entered;
        int id = 0;
        double timestamp = 0.0;
    };

    // ByteTrack (Zhang et al., 2022): IoU-based multi-object tracker with a
    // Kalman motion model and a second association pass for low-score detections.
    // No appearance features; a ReID embedding can be added to the cost later.
    //
    // A new track is confirmed (gets an ID and an Entered event) when it is matched
    // again in the next frame; detections in the very first frame are confirmed
    // immediately. A confirmed track that stays lost longer than lostTimeout is
    // removed with a Left event.
    class ByteTracker
    {

    public:

        explicit ByteTracker(TrackerConfig config = TrackerConfig());

        // Call once per frame, also with an empty list when nobody is detected.
        // timestamp in seconds (the image stamp). Time going backwards (a rosbag
        // restarting) resets the tracker.
        const std::vector<Track>& update(const std::vector<Detection>& detections, double timestamp);

        // Confirmed tracks matched in the last update, sorted by ID.
        const std::vector<Track>& tracks() const { return _output; }
        // Entered / Left events of the last update.
        const std::vector<TrackEvent>& events() const { return _events; }
        // Confirmed tracks currently waiting for their person to reappear.
        size_t lostCount() const;

        // Drops all tracks without events. IDs keep counting.
        void reset();

        const TrackerConfig& config() const { return _config; }

    private:

        enum class TrackState { Tentative, Tracked, Lost, Removed };

        struct TrackData
        {
            TrackData(const Detection& detection, int index, double timestamp);

            KalmanBoxFilter filter;
            TrackState state = TrackState::Tentative;
            int id = 0;     // 0 until confirmed
            float score = 0.0F;
            int detectionIndex = -1;
            double firstSeen = 0.0;
            double lastSeen = 0.0;
        };

        struct Association
        {
            std::vector<std::pair<size_t, size_t>> matches;     // (track, detection)
            std::vector<size_t> unmatchedTracks;
            std::vector<size_t> unmatchedDetections;
        };

        Association associate(const std::vector<size_t>& trackIndices, const std::vector<size_t>& detectionIndices,
                               const std::vector<Detection>& detections, float threshold, bool fuseScore) const;

        void applyMatch(TrackData& track, const std::vector<Detection>& detections, size_t index, double timestamp);
        void confirm(TrackData& track, double timestamp);
        void remove(TrackData& track, double timestamp);
        void removeDuplicates(double timestamp);
        void buildOutput();

    private:

        TrackerConfig _config;
        std::vector<TrackData> _tracks;
        std::vector<Track> _output;
        std::vector<TrackEvent> _events;

        int _nextId = 1;
        bool _hasTimestamp = false;
        double _lastTimestamp = 0.0;
        bool _firstFrame = true;
    };
}
