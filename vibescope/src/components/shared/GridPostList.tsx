import { Models } from "appwrite";
import { Link } from "react-router-dom";

import { useUserContext } from "@/context/AuthContext";
import PostStats from "./PostStats";

type GridPostListProps = {
  posts: Models.Document[];
  showUser?: boolean;
  showStats?: boolean;
};

const GridPostList = ({
  posts,
  showUser = true,
  showStats = true,
}: GridPostListProps) => {
  const { user } = useUserContext();

  return (
    <ul className="grid-container">
      {posts.map((post) => (
        <li key={post.$id} className="relative min-w-80 h-80">
          <Link to={`/posts/${post.$id}`} className="grid-post_link">
            <img
              src={post.imageUrl || "/assets/icons/post-placeholder.svg"} // Fallback for post image
              alt="post"
              className="h-full w-full object-cover"
            />
          </Link>

          <div className="grid-post_user">
            {showUser && (
              <div className="flex items-center justify-start gap-2 flex-1">
                <img
                  src={
                    post.creator?.imageUrl || // Ensure post.creator exists
                    "/assets/icons/profile-placeholder.svg" // Fallback for creator image
                  }
                  alt="creator"
                  className="w-8 h-8 rounded-full"
                />
                <p className="line-clamp-1">
                  {post.creator?.name || "Unknown User"} {/* Fallback for creator name */}
                </p>
              </div>
            )}
            {showStats && <PostStats post={post} userId={user.id} />}
          </div>
        </li>
      ))}
    </ul>
  );
};

export default GridPostList