#include <gtest/gtest.h>

#include "meldog_perception/tracking/byte_tracker.hpp"

using meldog_perception::Detection;
using meldog_perception::tracking::ByteTracker;
using meldog_perception::tracking::Track;
using meldog_perception::tracking::TrackEvent;

namespace {

    constexpr double kFrame = 1.0 / 30.0;

    Detection person(float x, float y, float score = 0.9F, float w = 50.0F, float h = 120.0F)
    {
        return {cv::Rect2f(x, y, w, h), score, 0};
    }

    // Runs the tracker and collects all events, so tests can check the whole history.
    class TrackerRun
    {

    public:

        const std::vector<Track>& step(const std::vector<Detection>& detections)
        {
            const auto& tracks = tracker.update(detections, frame * kFrame);
            ++frame;
            events.insert(events.end(), tracker.events().begin(), tracker.events().end());
            return tracks;
        }

        int count(TrackEvent::Type type) const
        {
            int n = 0;
            for (const auto& event : events) {
                n += event.type == type ? 1 : 0;
            }
            return n;
        }

        ByteTracker tracker;
        std::vector<TrackEvent> events;
        int frame = 0;
    };
}

TEST(ByteTracker, KeepsIdOfAMovingPerson)
{
    TrackerRun run;
    for (int i = 0; i < 60; ++i) {
        const auto& tracks = run.step({person(100.0F + 5.0F * i, 200.0F)});
        ASSERT_EQ(tracks.size(), 1U) << "frame " << i;
        EXPECT_EQ(tracks[0].id, 1);
        EXPECT_EQ(tracks[0].detectionIndex, 0);
    }
    EXPECT_EQ(run.count(TrackEvent::Type::Entered), 1);
    EXPECT_EQ(run.count(TrackEvent::Type::Left), 0);

    // 5 px per frame at 30 FPS.
    EXPECT_NEAR(run.tracker.tracks()[0].velocity.x, 150.0F, 20.0F);
}

TEST(ByteTracker, KeepsIdThroughAShortOcclusion)
{
    TrackerRun run;
    for (int i = 0; i < 20; ++i) {
        run.step({person(100.0F + 5.0F * i, 200.0F)});
    }
    for (int i = 20; i < 35; ++i) {   // hidden for 0.5 s, still walking
        EXPECT_TRUE(run.step({}).empty());
    }
    EXPECT_EQ(run.tracker.lostCount(), 1U);

    const auto& tracks = run.step({person(100.0F + 5.0F * 35, 200.0F)});
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].id, 1);
    EXPECT_EQ(run.count(TrackEvent::Type::Entered), 1);
    EXPECT_EQ(run.count(TrackEvent::Type::Left), 0);
}

TEST(ByteTracker, GivesANewIdAfterTheLostTimeout)
{
    TrackerRun run;
    for (int i = 0; i < 10; ++i) {
        run.step({person(100.0F, 200.0F)});
    }
    for (int i = 0; i < 45; ++i) {    // 1.5 s > lostTimeout
        run.step({});
    }
    EXPECT_EQ(run.count(TrackEvent::Type::Left), 1);
    EXPECT_EQ(run.tracker.lostCount(), 0U);

    // A returning person needs two frames to be confirmed.
    EXPECT_TRUE(run.step({person(100.0F, 200.0F)}).empty());
    const auto& tracks = run.step({person(100.0F, 200.0F)});
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].id, 2);
    EXPECT_EQ(run.count(TrackEvent::Type::Entered), 2);
}

TEST(ByteTracker, LowScoreDetectionsKeepTheTrack)
{
    TrackerRun run;
    for (int i = 0; i < 10; ++i) {
        run.step({person(100.0F, 200.0F)});
    }
    for (int i = 0; i < 15; ++i) {    // partly occluded: the detector is unsure
        const auto& tracks = run.step({person(100.0F, 200.0F, 0.3F)});
        ASSERT_EQ(tracks.size(), 1U) << "frame " << i;
        EXPECT_EQ(tracks[0].id, 1);
    }
}

TEST(ByteTracker, IgnoresDetectionsBelowTheLowThreshold)
{
    TrackerRun run;
    for (int i = 0; i < 10; ++i) {
        run.step({person(100.0F, 200.0F)});
    }
    EXPECT_TRUE(run.step({person(100.0F, 200.0F, 0.05F)}).empty());
    EXPECT_EQ(run.tracker.lostCount(), 1U);
}

TEST(ByteTracker, SingleFrameFalsePositiveIsNotConfirmed)
{
    TrackerRun run;
    for (int i = 0; i < 10; ++i) {
        run.step({person(100.0F, 200.0F)});
    }
    const auto& withGhost = run.step({person(100.0F, 200.0F), person(400.0F, 100.0F)});
    EXPECT_EQ(withGhost.size(), 1U);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(run.step({person(100.0F, 200.0F)}).size(), 1U);
    }
    EXPECT_EQ(run.count(TrackEvent::Type::Entered), 1);
    EXPECT_EQ(run.count(TrackEvent::Type::Left), 0);
}

TEST(ByteTracker, TwoPeopleCrossingKeepTheirIds)
{
    TrackerRun run;
    int idA = 0;
    int idB = 0;
    for (int i = 0; i < 60; ++i) {
        // A walks right, B walks left, slightly lower and taller; they pass at frame 30.
        const Detection a = person(100.0F + 5.0F * i, 200.0F, 0.9F, 50.0F, 120.0F);
        const Detection b = person(400.0F - 5.0F * i, 220.0F, 0.85F, 55.0F, 140.0F);
        const auto& tracks = run.step({a, b});
        ASSERT_EQ(tracks.size(), 2U) << "frame " << i;

        for (const auto& track : tracks) {
            const bool isA = track.detectionIndex == 0;
            int& expected = isA ? idA : idB;
            if (expected == 0) {
                expected = track.id;
            }
            EXPECT_EQ(track.id, expected) << "frame " << i << (isA ? " person A" : " person B");
        }
    }
    EXPECT_NE(idA, idB);
}

TEST(ByteTracker, TimeGoingBackwardsResets)
{
    ByteTracker tracker;
    tracker.update({person(100.0F, 200.0F)}, 10.0);
    ASSERT_EQ(tracker.tracks().size(), 1U);

    const auto& tracks = tracker.update({person(300.0F, 200.0F)}, 5.0);
    ASSERT_EQ(tracker.events().size(), 2U);
    EXPECT_EQ(tracker.events()[0].type, TrackEvent::Type::Left);
    EXPECT_EQ(tracker.events()[1].type, TrackEvent::Type::Entered);
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].id, 2);
}

TEST(ByteTracker, HandlesAnIrregularFrameRate)
{
    // Detector slower than the camera: 10 FPS instead of 30.
    ByteTracker tracker;
    for (int i = 0; i < 30; ++i) {
        const auto& tracks = tracker.update({person(100.0F + 15.0F * i, 200.0F)}, i * 0.1);
        ASSERT_EQ(tracks.size(), 1U) << "frame " << i;
        EXPECT_EQ(tracks[0].id, 1);
    }
    EXPECT_NEAR(tracker.tracks()[0].velocity.x, 150.0F, 20.0F);
}
