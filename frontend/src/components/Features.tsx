import { Card } from "@/components/ui/card";
import { Brain, Headphones, Settings, Users, Trophy, Heart } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Advanced algorithms adapt to your reading patterns and provide personalized support.",
    gradient: "from-primary/20 to-primary/5",
  },
  {
    icon: Headphones,
    title: "Natural Voice Reading",
    description: "High-quality text-to-speech with natural intonation helps follow along effortlessly.",
    gradient: "from-secondary/20 to-secondary/5",
  },
  {
    icon: Settings,
    title: "Fully Customizable",
    description: "Adjust fonts, colors, spacing, and more to create your perfect reading environment.",
    gradient: "from-accent/20 to-accent/5",
  },
  {
    icon: Users,
    title: "Teacher Dashboard",
    description: "Track student progress and provide targeted support where it's needed most.",
    gradient: "from-info/20 to-info/5",
  },
  {
    icon: Trophy,
    title: "Progress Tracking",
    description: "Celebrate achievements and build confidence with milestone tracking and rewards.",
    gradient: "from-success/20 to-success/5",
  },
  {
    icon: Heart,
    title: "Built with Care",
    description: "Designed with input from educators, parents, and learners with dyslexia.",
    gradient: "from-primary/20 to-accent/5",
  },
];

const Features = () => {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-6xl">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-3xl font-bold md:text-4xl">
              Features That Make a Difference
            </h2>
            <p className="text-lg text-muted-foreground">
              Everything you need to support confident, successful readers
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card
                  key={index}
                  className="group p-6 shadow-soft transition-all duration-300 hover:shadow-medium"
                >
                  <div className={`mb-4 inline-flex rounded-xl bg-gradient-to-br ${feature.gradient} p-3`}>
                    <Icon className="h-6 w-6 text-foreground" />
                  </div>
                  <h3 className="mb-2 text-xl font-semibold">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </Card>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
